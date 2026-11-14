# Experiment 11 - model recovery on the ladder (SYNTHETIC data)

20 datasets per generator, 150 trials, 4 options.


## BIC: overall accuracy over generators that are candidates = 0.58 (chance 0.14)

P(selected | generating):

| generating | random | wsls | q | asym_q | sticky_q | lapse_q | kalman |
|---|---|---|---|---|---|---|---|
| random | 0.95 | 0.05 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| wsls | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| q | 0.15 | 0.0 | 0.65 | 0.0 | 0.0 | 0.0 | 0.2 |
| asym_q | 0.25 | 0.0 | 0.6 | 0.05 | 0.05 | 0.0 | 0.05 |
| sticky_q | 0.0 | 0.3 | 0.1 | 0.0 | 0.55 | 0.0 | 0.05 |
| lapse_q | 0.15 | 0.0 | 0.7 | 0.05 | 0.0 | 0.05 | 0.05 |
| kalman | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 0.8 |
| full_q (mis-specified) | 0.0 | 0.2 | 0.15 | 0.0 | 0.65 | 0.0 | 0.0 |


P(generating | selected):

| generating | random | wsls | q | asym_q | sticky_q | lapse_q | kalman |
|---|---|---|---|---|---|---|---|
| random | 0.63 | 0.03 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| wsls | 0.0 | 0.65 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| q | 0.1 | 0.0 | 0.27 | 0.0 | 0.0 | 0.0 | 0.17 |
| asym_q | 0.17 | 0.0 | 0.25 | 0.5 | 0.04 | 0.0 | 0.04 |
| sticky_q | 0.0 | 0.19 | 0.04 | 0.0 | 0.44 | 0.0 | 0.04 |
| lapse_q | 0.1 | 0.0 | 0.29 | 0.5 | 0.0 | 1.0 | 0.04 |
| kalman | 0.0 | 0.0 | 0.08 | 0.0 | 0.0 | 0.0 | 0.7 |
| full_q (mis-specified) | 0.0 | 0.13 | 0.06 | 0.0 | 0.52 | 0.0 | 0.0 |


Per-generator accuracy with Wilson 95% interval:

| criterion | generating | n_datasets | n_correct | accuracy | ci_lo | ci_hi |
|---|---|---|---|---|---|---|
| bic | random | 20 | 19 | 0.95 | 0.764 | 0.991 |
| bic | wsls | 20 | 20 | 1.0 | 0.839 | 1.0 |
| bic | q | 20 | 13 | 0.65 | 0.433 | 0.819 |
| bic | asym_q | 20 | 1 | 0.05 | 0.009 | 0.236 |
| bic | sticky_q | 20 | 11 | 0.55 | 0.342 | 0.742 |
| bic | lapse_q | 20 | 1 | 0.05 | 0.009 | 0.236 |
| bic | kalman | 20 | 16 | 0.8 | 0.584 | 0.919 |
| bic | full_q (mis-specified) | 20 | 0 | 0.0 | 0.0 | 0.161 |



## AIC: overall accuracy over generators that are candidates = 0.60 (chance 0.14)

P(selected | generating):

| generating | random | wsls | q | asym_q | sticky_q | lapse_q | kalman |
|---|---|---|---|---|---|---|---|
| random | 0.8 | 0.15 | 0.0 | 0.0 | 0.05 | 0.0 | 0.0 |
| wsls | 0.0 | 0.9 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 |
| q | 0.1 | 0.0 | 0.55 | 0.0 | 0.0 | 0.05 | 0.3 |
| asym_q | 0.2 | 0.0 | 0.45 | 0.15 | 0.1 | 0.0 | 0.1 |
| sticky_q | 0.0 | 0.2 | 0.05 | 0.0 | 0.7 | 0.0 | 0.05 |
| lapse_q | 0.0 | 0.05 | 0.4 | 0.1 | 0.2 | 0.1 | 0.15 |
| kalman | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| full_q (mis-specified) | 0.0 | 0.15 | 0.0 | 0.1 | 0.7 | 0.05 | 0.0 |


P(generating | selected):

| generating | random | wsls | q | asym_q | sticky_q | lapse_q | kalman |
|---|---|---|---|---|---|---|---|
| random | 0.73 | 0.1 | 0.0 | 0.0 | 0.03 | 0.0 | 0.0 |
| wsls | 0.0 | 0.62 | 0.0 | 0.0 | 0.05 | 0.0 | 0.0 |
| q | 0.09 | 0.0 | 0.38 | 0.0 | 0.0 | 0.25 | 0.19 |
| asym_q | 0.18 | 0.0 | 0.31 | 0.43 | 0.05 | 0.0 | 0.06 |
| sticky_q | 0.0 | 0.14 | 0.03 | 0.0 | 0.38 | 0.0 | 0.03 |
| lapse_q | 0.0 | 0.03 | 0.28 | 0.29 | 0.11 | 0.5 | 0.09 |
| kalman | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.62 |
| full_q (mis-specified) | 0.0 | 0.1 | 0.0 | 0.29 | 0.38 | 0.25 | 0.0 |


Per-generator accuracy with Wilson 95% interval:

| criterion | generating | n_datasets | n_correct | accuracy | ci_lo | ci_hi |
|---|---|---|---|---|---|---|
| aic | random | 20 | 16 | 0.8 | 0.584 | 0.919 |
| aic | wsls | 20 | 18 | 0.9 | 0.699 | 0.972 |
| aic | q | 20 | 11 | 0.55 | 0.342 | 0.742 |
| aic | asym_q | 20 | 3 | 0.15 | 0.052 | 0.36 |
| aic | sticky_q | 20 | 14 | 0.7 | 0.481 | 0.855 |
| aic | lapse_q | 20 | 2 | 0.1 | 0.028 | 0.301 |
| aic | kalman | 20 | 20 | 1.0 | 0.839 | 1.0 |
| aic | full_q (mis-specified) | 20 | 0 | 0.0 | 0.0 | 0.161 |


