# Experiments 6 & 7 - model comparison and prediction (SYNTHETIC data)

Population: 10 Q-learners, 10 POMDP trackers, 10 BAMDP planners; 200 trials each.

## Summary over participants

| model | total_log_likelihood | total_aic | total_bic | mean_aic_weight | n_wins_aic | n_wins_bic |
|---|---|---|---|---|---|---|
| q_learning | -2941.19 | 6002.38 | 6200.28 | 0.21 | 8 | 10 |
| asym_q | -2733.56 | 5647.13 | 5943.98 | 0.14 | 2 | 0 |
| bamdp | -3714.17 | 7488.34 | 7587.29 | 0.31 | 10 | 10 |
| pomdp | -2928.87 | 5977.74 | 6175.64 | 0.34 | 10 | 10 |


## Winner by generating model (counts)

| generating_model | model | AIC | BIC |
|---|---|---|---|
| bamdp | bamdp | 10 | 10 |
| pomdp | pomdp | 10 | 10 |
| q_learning | asym_q | 2 | 0 |
| q_learning | q_learning | 8 | 10 |


## Held-out prediction, pooled over the mixed population (hides who wins for whom)

| train_fraction | model | test_ll_per_trial | test_accuracy |
|---|---|---|---|
| 0.3 | asym_q | -0.637 | 0.72 |
| 0.3 | bamdp | -0.66 | 0.621 |
| 0.3 | chance | -0.693 | 0.5 |
| 0.3 | pomdp | -0.532 | 0.712 |
| 0.3 | q_learning | -0.766 | 0.719 |
| 0.5 | asym_q | -0.539 | 0.741 |
| 0.5 | bamdp | -0.651 | 0.627 |
| 0.5 | chance | -0.693 | 0.5 |
| 0.5 | pomdp | -0.503 | 0.725 |
| 0.5 | q_learning | -0.58 | 0.7 |
| 0.7 | asym_q | -0.528 | 0.751 |
| 0.7 | bamdp | -0.632 | 0.656 |
| 0.7 | chance | -0.693 | 0.5 |
| 0.7 | pomdp | -0.49 | 0.731 |
| 0.7 | q_learning | -0.53 | 0.701 |


## Held-out prediction by generating model

| train_fraction | generating_model | model | test_ll_per_trial | test_accuracy |
|---|---|---|---|---|
| 0.3 | bamdp | asym_q | -1.023 | 0.56 |
| 0.3 | bamdp | bamdp | -0.621 | 0.666 |
| 0.3 | bamdp | pomdp | -0.744 | 0.504 |
| 0.3 | bamdp | q_learning | -1.443 | 0.591 |
| 0.3 | pomdp | asym_q | -0.488 | 0.776 |
| 0.3 | pomdp | bamdp | -0.685 | 0.593 |
| 0.3 | pomdp | pomdp | -0.403 | 0.834 |
| 0.3 | pomdp | q_learning | -0.474 | 0.74 |
| 0.3 | q_learning | asym_q | -0.4 | 0.822 |
| 0.3 | q_learning | bamdp | -0.673 | 0.605 |
| 0.3 | q_learning | pomdp | -0.45 | 0.8 |
| 0.3 | q_learning | q_learning | -0.381 | 0.825 |
| 0.5 | bamdp | asym_q | -0.732 | 0.617 |
| 0.5 | bamdp | bamdp | -0.615 | 0.655 |
| 0.5 | bamdp | pomdp | -0.71 | 0.53 |
| 0.5 | bamdp | q_learning | -0.842 | 0.53 |
| 0.5 | pomdp | asym_q | -0.509 | 0.781 |
| 0.5 | pomdp | bamdp | -0.672 | 0.608 |
| 0.5 | pomdp | pomdp | -0.36 | 0.838 |
| 0.5 | pomdp | q_learning | -0.525 | 0.748 |
| 0.5 | q_learning | asym_q | -0.376 | 0.826 |
| 0.5 | q_learning | bamdp | -0.667 | 0.619 |
| 0.5 | q_learning | pomdp | -0.437 | 0.807 |
| 0.5 | q_learning | q_learning | -0.373 | 0.823 |
| 0.7 | bamdp | asym_q | -0.699 | 0.625 |
| 0.7 | bamdp | bamdp | -0.591 | 0.668 |
| 0.7 | bamdp | pomdp | -0.73 | 0.505 |
| 0.7 | bamdp | q_learning | -0.725 | 0.53 |
| 0.7 | pomdp | asym_q | -0.552 | 0.785 |
| 0.7 | pomdp | bamdp | -0.656 | 0.652 |
| 0.7 | pomdp | pomdp | -0.357 | 0.85 |
| 0.7 | pomdp | q_learning | -0.532 | 0.73 |
| 0.7 | q_learning | asym_q | -0.333 | 0.842 |
| 0.7 | q_learning | bamdp | -0.65 | 0.648 |
| 0.7 | q_learning | pomdp | -0.383 | 0.838 |
| 0.7 | q_learning | q_learning | -0.333 | 0.842 |


## Best held-out model for each generating model

| train_fraction | generating_model | best_held_out_model |
|---|---|---|
| 0.3 | bamdp | bamdp |
| 0.3 | pomdp | pomdp |
| 0.3 | q_learning | q_learning |
| 0.5 | bamdp | bamdp |
| 0.5 | pomdp | pomdp |
| 0.5 | q_learning | q_learning |
| 0.7 | bamdp | bamdp |
| 0.7 | pomdp | pomdp |
| 0.7 | q_learning | q_learning |
