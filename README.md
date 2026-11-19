# Toy prototype results — federated safe MARL, microgrids with dropouts

IMPORTANT: this is a NumPy toy simulation (synthetic load/PV/price, shared feeder limit), built to check the pipeline and shape the real experiment. These numbers are NOT research results and must not appear on the CV. Re-run the study in PowerGridworld/pandapower before reporting anything.

Setup: N microgrids, local-only observations, battery charge/discharge actions, shared feeder import limit (0.92 x total mean load), per-agent Gaussian policy trained with REINFORCE + Lagrangian constraint (dual variable per agent). Federation: server aggregates policy parameters; uploads and downloads dropped independently with probability p. Methods: independent; FedAvg (average what arrives); CSW (cache last received models, staleness-weighted average, also averages dual variables). 5 seeds, mean ± std.

## Findings
1. Constraint handling works in the toy: idle battery 22.9% of time steps overloaded; learners ignoring the constraint ~11%; constraint-aware independent learners ~5%, at about 3% higher energy cost (4.02 vs 3.90 $/agent/day). Federated safe runs at N=5 had large seed variance (7.6 ± 8.5%).
2. With 2 local steps per round, dropout barely mattered: averaging every round keeps agents nearly identical.
3. With 10 local steps per round (more drift between syncs), FedAvg degraded at heavy dropout: 50% dropout gave 12.0 ± 9.9% overloaded steps and cost 4.12 vs CSW 4.3 ± 1.4% and 3.96; 70%: 9.4 ± 7.8% vs 3.6 ± 1.7%. With 5 seeds this is suggestive, not conclusive.
4. N=10: feeder limit scales with N and rarely binds (<1% violations for all methods) — uninformative; needs a tighter limit.

## Caveats
- Environment limit factor and dual target were tuned after seeing early runs; a limit factor of 0.88 was too tight (constraint-aware learners did worse than selfish ones).
- CSW is a simple candidate rule, not a validated method.
- Second table (10 local steps) was run after seeing the first table showed no dropout effect.

