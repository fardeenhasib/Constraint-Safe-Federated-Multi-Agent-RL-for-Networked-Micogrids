"""
Toy prototype: constraint-safe federated multi-agent RL for networked microgrids
under communication dropouts.  NumPy only (manual backprop), CPU-fast.

THIS IS A TOY SIMULATION for pipeline checking -- not a validated power-system
model.  Do not report its numbers as research results.

Setup
-----
* N microgrids, each with load, PV, and a battery.  Each agent sees only LOCAL
  observations (partial observability) and picks a battery charge/discharge power.
* Coupling: all microgrids share one feeder with an aggregate import limit L.
  A time-of-use price makes every agent want to charge at the same cheap hours,
  which overloads the feeder unless they coordinate implicitly through the
  Lagrangian-penalised shared violation signal.
* Training: per-agent Gaussian policy, REINFORCE with time-dependent baseline,
  Lagrangian constraint handling (per-agent dual variable).
* Federation: every round each agent trains locally, then a server aggregates
  policy parameters.  Uploads and downloads are dropped independently with prob p.

Methods
-------
independent : no communication (lower bound for coordination, upper bound for privacy)
fedavg      : average whatever arrives this round, broadcast to whoever is reachable
csw         : cache + staleness-weighted aggregation, also averages dual variables
              (candidate dropout-robust aggregation rule for the real study)
"""
import numpy as np

D_OBS, H = 5, 16
T = 24
PRICE = np.array([0.10] * 6 + [0.20] * 11 + [0.40] * 5 + [0.20] * 2)  # $/kWh, 24h
SELL = 0.3  # export is paid at SELL * price


# ----------------------------------------------------------------------------- env
class MicrogridEnv:
    def __init__(self, n_agents, rng, limit_factor=0.92):
        self.N = n_agents
        r = np.random.default_rng(1234 + n_agents)  # fixed fleet heterogeneity
        self.load_scale = r.uniform(0.7, 1.3, n_agents)
        self.pv_cap = r.uniform(0.6, 1.6, n_agents)
        self.pmax = r.uniform(0.5, 0.9, n_agents)
        self.ecap = r.uniform(1.5, 3.0, n_agents)
        self.limit = limit_factor * np.sum(self.load_scale * 0.95)  # feeder limit (kW)
        self.rng = rng

    def reset(self, B):
        N, rng = self.N, self.rng
        self.B = B
        self.soc = rng.uniform(0.3, 0.7, (B, N))
        self.cloud_shared = rng.uniform(0.5, 1.0, (B, 1))
        self.cloud_ind = rng.uniform(0.85, 1.0, (B, N))
        self.t = 0
        return self._obs()

    def _load(self, t):
        base = 0.9 + 0.3 * np.sin(2 * np.pi * (t - 7) / 24) + 0.15 * (17 <= t < 22)
        return base * self.load_scale * (1 + 0.05 * self.rng.standard_normal((self.B, self.N)))

    def _pv(self, t):
        shape = max(0.0, np.sin(np.pi * (t - 6) / 12)) if 6 <= t <= 18 else 0.0
        return self.pv_cap * shape * self.cloud_shared * self.cloud_ind

    def _obs(self):
        t = min(self.t, T - 1)
        self.cur_load, self.cur_pv = self._load(t), self._pv(t)
        o = np.empty((self.B, self.N, D_OBS))
        o[..., 0] = t / T
        o[..., 1] = self.cur_load / self.load_scale
        o[..., 2] = self.cur_pv / self.pv_cap
        o[..., 3] = self.soc
        o[..., 4] = PRICE[t] / 0.4
        return o

    def step(self, a):
        """a in [-1,1]^(B,N). Returns obs, reward(B,N), shared cost(B), violated(B), done."""
        a = np.clip(a, -1.0, 1.0)
        p = a * self.pmax  # kW, + = charge
        eff = 0.95
        dsoc = np.where(p >= 0, p * eff, p / eff) / self.ecap
        new_soc = np.clip(self.soc + dsoc, 0.1, 0.9)
        dsoc_real = new_soc - self.soc
        p_real = np.where(dsoc_real >= 0, dsoc_real * self.ecap / eff, dsoc_real * self.ecap * eff)
        self.soc = new_soc
        imp = self.cur_load - self.cur_pv + p_real  # kW, + = import
        price = PRICE[self.t]
        money = np.where(imp > 0, price * imp, SELL * price * imp)  # $
        reward = -money
        S = imp.sum(1)
        viol = np.maximum(0.0, S - self.limit) / self.N  # normalised overload
        self.t += 1
        done = self.t >= T
        obs = None if done else self._obs()
        return obs, reward, viol, money, done


# -------------------------------------------------------------------- policy params
def init_params(N, rng):
    return {
        "W1": rng.standard_normal((N, D_OBS, H)) * 0.3,
        "b1": np.zeros((N, H)),
        "W2": rng.standard_normal((N, H)) * 0.1,
        "b2": np.zeros(N),
        "ls": np.full(N, np.log(0.5)),
    }


def forward(P, X):
    h = np.tanh(np.einsum("bnd,ndh->bnh", X, P["W1"]) + P["b1"])
    mu = np.einsum("bnh,nh->bn", h, P["W2"]) + P["b2"]
    return h, mu


class Adam:
    def __init__(self, P, lr=3e-3):
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t, self.lr = 0, lr

    def step(self, P, G):
        self.t += 1
        for k in P:
            self.m[k] = 0.9 * self.m[k] + 0.1 * G[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * G[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def rollout(env, P, B, rng, deterministic=False):
    obs = env.reset(B)
    Xs, hs, eps_l, rew, cost, viol_l, money_l = [], [], [], [], [], [], []
    std = np.exp(P["ls"])
    for _ in range(T):
        h, mu = forward(P, obs)
        eps = np.zeros_like(mu) if deterministic else rng.standard_normal(mu.shape)
        a = mu + std * eps
        nobs, r, v, money, done = env.step(a)
        Xs.append(obs); hs.append(h); eps_l.append(eps)
        rew.append(r); viol_l.append(v); money_l.append(money)
        obs = nobs
    return (np.array(Xs), np.array(hs), np.array(eps_l), np.array(rew),
            np.array(viol_l), np.array(money_l))


def local_update(env, P, opt, lam, B, rng, d_target=0.0005, ent=1e-3, eta_lam=2.0):
    X, hs, eps, rew, viol, _ = rollout(env, P, B, rng)
    # Lagrangian-adjusted reward (shared violation signal, per-agent multiplier)
    r_adj = rew - lam[None, None, :] * viol[:, :, None] * 10.0
    G = np.cumsum(r_adj[::-1], axis=0)[::-1]  # undiscounted return-to-go, (T,B,N)
    A = G - G.mean(1, keepdims=True)  # time-dependent baseline
    A = A / (A.std(axis=(0, 1), keepdims=True) + 1e-8)
    std = np.exp(P["ls"])
    n = T * B
    dmu = -A * eps / std / n  # (T,B,N)
    g = {}
    g["W2"] = np.einsum("tbn,tbnh->nh", dmu, hs)
    g["b2"] = dmu.sum((0, 1))
    dz = (dmu[..., None] * P["W2"][None, None]) * (1 - hs ** 2)
    g["W1"] = np.einsum("tbnd,tbnh->ndh", X, dz)
    g["b1"] = dz.sum((0, 1))
    g["ls"] = (-A * (eps ** 2 - 1)).sum((0, 1)) / n - ent
    opt.step(P, g)
    # dual ascent on the average per-step violation
    lam += eta_lam * (viol.mean() - d_target)
    np.maximum(lam, 0.0, out=lam)


# ------------------------------------------------------------------- federation
def aggregate(method, P, lam, p_drop, rng, cache):
    """Return number of successful uploads this round. Modifies P, lam in place."""
    N = P["W2"].shape[0]
    if method == "independent":
        return 0
    up = rng.random(N) > p_drop
    down = rng.random(N) > p_drop
    n_up = int(up.sum())
    keys = list(P.keys())
    if method == "fedavg":
        if n_up == 0:
            return 0
        glob = {k: P[k][up].mean(0) for k in keys}
        for k in keys:
            P[k][down] = glob[k]
        return n_up
    if method == "csw":
        # server keeps the last received model + age of every client
        for i in np.where(up)[0]:
            for k in keys:
                cache["P"][k][i] = P[k][i]
            cache["lam"][i] = lam[i]
            cache["age"][i] = 0
        cache["age"][~up] += 1
        if cache["init"] is False and n_up == 0:
            return 0
        cache["init"] = True
        w = np.exp(-cache["age"] / 3.0)
        w = w / w.sum()
        glob = {k: np.tensordot(w, cache["P"][k], axes=(0, 0)) for k in keys}
        glob_lam = float(np.dot(w, cache["lam"]))
        for k in keys:
            P[k][down] = glob[k]
        lam[down] = glob_lam
        return n_up
    raise ValueError(method)


def evaluate(env, P, rng, n_ep=200):
    X, hs, eps, rew, viol, money = rollout(env, P, n_ep, rng, deterministic=True)
    return {
        "viol_rate": float((viol > 1e-9).mean()),  # fraction of timesteps overloaded
        "viol_mag": float(viol.mean()),  # mean normalised overload per step
        "cost": float(money.sum(0).mean()),  # $/agent/day
    }


def run(method, n_agents=5, p_drop=0.0, seed=0, rounds=200, local_steps=2, B=32, safe=True):
    rng = np.random.default_rng(seed)
    env = MicrogridEnv(n_agents, np.random.default_rng(seed + 10_000))
    P = init_params(n_agents, rng)
    # all clients start from one common model (standard FL initialisation)
    for k in P:
        P[k][:] = P[k][0]
    opt = Adam(P)
    lam = np.zeros(n_agents)
    cache = {"P": {k: v.copy() for k, v in P.items()}, "lam": lam.copy(),
             "age": np.zeros(n_agents), "init": False}
    uploads, hist = 0, []
    ev_env = MicrogridEnv(n_agents, np.random.default_rng(seed + 20_000))
    for r in range(rounds):
        for _ in range(local_steps):
            local_update(env, P, opt, lam, B, rng, eta_lam=2.0 if safe else 0.0)
        uploads += aggregate(method, P, lam, p_drop, rng, cache)
        if r >= rounds - 20:  # last 20 rounds -> stability measure
            hist.append(evaluate(ev_env, P, np.random.default_rng(seed + 30_000 + r), 50))
    final = evaluate(ev_env, P, np.random.default_rng(seed + 99_999), 300)
    final["uploads"] = uploads
    final["viol_mag_std_last20"] = float(np.std([h["viol_mag"] for h in hist]))
    # unconstrained-greedy reference: how bad is the coupling without coordination?
    return final


if __name__ == "__main__":
    import time
    t0 = time.time()
    print(run("fedavg", p_drop=0.1, seed=0), f"{time.time()-t0:.1f}s")
