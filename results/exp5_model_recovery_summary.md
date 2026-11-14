# Experiment 5 - model recovery (SYNTHETIC data)


## BIC, 100 trials per dataset - accuracy 0.67 (chance 0.25)

Confusion matrix P(selected | generating):

| generating | q_learning | asym_q | bamdp | pomdp |
|---|---|---|---|---|
| q_learning | 0.7 | 0.0 | 0.13 | 0.17 |
| asym_q | 0.6 | 0.13 | 0.03 | 0.23 |
| bamdp | 0.03 | 0.0 | 0.97 | 0.0 |
| pomdp | 0.1 | 0.0 | 0.03 | 0.87 |


Inversion matrix P(generating | selected):

| generating | q_learning | asym_q | bamdp | pomdp |
|---|---|---|---|---|
| q_learning | 0.49 | 0.0 | 0.11 | 0.13 |
| asym_q | 0.42 | 1.0 | 0.03 | 0.18 |
| bamdp | 0.02 | 0.0 | 0.83 | 0.0 |
| pomdp | 0.07 | 0.0 | 0.03 | 0.68 |



## AIC, 100 trials per dataset - accuracy 0.68 (chance 0.25)

Confusion matrix P(selected | generating):

| generating | q_learning | asym_q | bamdp | pomdp |
|---|---|---|---|---|
| q_learning | 0.63 | 0.07 | 0.1 | 0.2 |
| asym_q | 0.43 | 0.33 | 0.0 | 0.23 |
| bamdp | 0.03 | 0.1 | 0.87 | 0.0 |
| pomdp | 0.07 | 0.03 | 0.03 | 0.87 |


Inversion matrix P(generating | selected):

| generating | q_learning | asym_q | bamdp | pomdp |
|---|---|---|---|---|
| q_learning | 0.54 | 0.12 | 0.1 | 0.15 |
| asym_q | 0.37 | 0.62 | 0.0 | 0.18 |
| bamdp | 0.03 | 0.19 | 0.87 | 0.0 |
| pomdp | 0.06 | 0.06 | 0.03 | 0.67 |



## BIC, 300 trials per dataset - accuracy 0.78 (chance 0.25)

Confusion matrix P(selected | generating):

| generating | q_learning | asym_q | bamdp | pomdp |
|---|---|---|---|---|
| q_learning | 0.93 | 0.0 | 0.03 | 0.03 |
| asym_q | 0.57 | 0.4 | 0.03 | 0.0 |
| bamdp | 0.0 | 0.0 | 1.0 | 0.0 |
| pomdp | 0.13 | 0.03 | 0.03 | 0.8 |


Inversion matrix P(generating | selected):

| generating | q_learning | asym_q | bamdp | pomdp |
|---|---|---|---|---|
| q_learning | 0.57 | 0.0 | 0.03 | 0.04 |
| asym_q | 0.35 | 0.92 | 0.03 | 0.0 |
| bamdp | 0.0 | 0.0 | 0.91 | 0.0 |
| pomdp | 0.08 | 0.08 | 0.03 | 0.96 |



## AIC, 300 trials per dataset - accuracy 0.82 (chance 0.25)

Confusion matrix P(selected | generating):

| generating | q_learning | asym_q | bamdp | pomdp |
|---|---|---|---|---|
| q_learning | 0.87 | 0.07 | 0.03 | 0.03 |
| asym_q | 0.37 | 0.63 | 0.0 | 0.0 |
| bamdp | 0.0 | 0.03 | 0.97 | 0.0 |
| pomdp | 0.07 | 0.1 | 0.03 | 0.8 |


Inversion matrix P(generating | selected):

| generating | q_learning | asym_q | bamdp | pomdp |
|---|---|---|---|---|
| q_learning | 0.67 | 0.08 | 0.03 | 0.04 |
| asym_q | 0.28 | 0.76 | 0.0 | 0.0 |
| bamdp | 0.0 | 0.04 | 0.94 | 0.0 |
| pomdp | 0.05 | 0.12 | 0.03 | 0.96 |


