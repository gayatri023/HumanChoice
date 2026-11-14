# Benchmarks (single process, 4 options, 150 trials, 3% timeouts)

| model | n_params | simulate_ms | likelihood_ms | fit_ms_4_starts | est_minutes_975_participants_one_fit |
|---|---|---|---|---|---|
| random | 0 | 5.39 | 0.96 | 0.94 | 0.02 |
| wsls | 2 | 4.22 | 1.29 | 147.65 | 2.4 |
| q | 2 | 3.55 | 0.08 | 13.32 | 0.22 |
| asym_q | 3 | 3.79 | 0.09 | 79.74 | 1.3 |
| sticky_q | 3 | 4.04 | 0.1 | 67.67 | 1.1 |
| lapse_q | 3 | 4.68 | 0.09 | 69.39 | 1.13 |
| kalman | 3 | 4.94 | 1.91 | 813.93 | 13.23 |

Whole ladder, one fit per participant for 975 participants: about 19 CPU-minutes; the full pipeline refits at 4 split points, so roughly 5x that, divided by `--jobs`.
