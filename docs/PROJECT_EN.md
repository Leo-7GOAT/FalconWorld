# FalconWorld

A reproducible research project for compact action-conditioned world models and receding-horizon control, with longitudinal following, synthetic Ackermann driving, CARLA adaptation, and an R550 deployment preparation path.

The pipeline generates episode-disjoint data, identifies strong physical dynamics from training data, fits compact neural/residual/history predictors, and evaluates recursive prediction and closed-loop CEM-MPC. Models share action limits, cost and safety supervision. Classic controllers, enriched physics, learned leader-only dynamics and uncertainty ablations remain visible.

This is model-based planning and control, not reinforcement learning. The synthetic longitudinal benchmark exposes actuator state; sensor-derived observations are evaluated separately. No future leader script enters a predictor. The enriched physical model can exactly identify the noise-free in-distribution ego mechanism, so a learned residual need not improve it.

## Current evidence

|Run|Status|Recorded / expected episodes|
|---|---|---|
|smoke_verified|verified_complete|304 / 304|
|standard_verified|verified_complete|4560 / 4560|

Incomplete results are provisional; smoke runs test the pipeline and cannot establish statistical superiority. The generated [Chinese report](REPORT.md) includes all model comparisons, episode-weighted recursive errors, paired effects when available, complete-cycle latency, training cost and deterministic failure replays. It retains adverse and null results.

Frozen software, data and seeds are reproducible; trajectories are not guaranteed bitwise identical because a measured wall-clock deadline triggers braking fallback, and host load can change that fallback. This does not demonstrate hard real-time behavior.

Run `python scripts/build_report.py` after experiments finish to refresh evidence. See the repository README for the verified reproduction commands, [CARLA](CARLA.md) for simulator acceptance, and [R550](R550.md) for replay, shadow mode, ARM64 preparation and hardware acceptance. Host export consistency and synthetic simulations do not establish Raspberry Pi timing or real-vehicle deployment. Real motion requires explicit authorization.

Generated: 2026-11-04T18:13:39+08:00.
