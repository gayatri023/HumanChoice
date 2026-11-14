# Experiment 10 - identifiability (SYNTHETIC data)

40 simulated participants per cell; 4 options unless stated; 3% timeouts; 4 optimiser starts (+ nested warm start).

## sticky_q: error correlations and leakage at T=150

Correlation of estimation errors:

| index | alpha | beta | kappa |
|---|---|---|---|
| alpha | 1.0 | -0.25 | -0.04 |
| beta | -0.25 | 1.0 | -0.1 |
| kappa | -0.04 | -0.1 | 1.0 |

Spearman(true parameter row, error in parameter column) - off-diagonal = leakage:

| index | alpha | beta | kappa |
|---|---|---|---|
| alpha | -0.04 | -0.16 | 0.24 |
| beta | 0.15 | -0.28 | -0.01 |
| kappa | -0.22 | 0.12 | 0.17 |

## lapse_q: error correlations and leakage at T=150

Correlation of estimation errors:

| index | alpha | beta | epsilon |
|---|---|---|---|
| alpha | 1.0 | -0.4 | -0.09 |
| beta | -0.4 | 1.0 | 0.31 |
| epsilon | -0.09 | 0.31 | 1.0 |

Spearman(true parameter row, error in parameter column) - off-diagonal = leakage:

| index | alpha | beta | epsilon |
|---|---|---|---|
| alpha | 0.03 | -0.51 | -0.33 |
| beta | -0.25 | -0.13 | 0.21 |
| epsilon | 0.25 | -0.38 | -0.49 |

## asym_q: error correlations and leakage at T=150

Correlation of estimation errors:

| index | alpha_pos | alpha_neg | beta |
|---|---|---|---|
| alpha_pos | 1.0 | 0.11 | -0.38 |
| alpha_neg | 0.11 | 1.0 | -0.41 |
| beta | -0.38 | -0.41 | 1.0 |

Spearman(true parameter row, error in parameter column) - off-diagonal = leakage:

| index | alpha_pos | alpha_neg | beta |
|---|---|---|---|
| alpha_pos | -0.28 | 0.06 | -0.08 |
| alpha_neg | 0.1 | -0.38 | 0.12 |
| beta | 0.2 | 0.21 | -0.37 |

## kalman: error correlations and leakage at T=150

Correlation of estimation errors:

| index | beta | phi | drift_ratio |
|---|---|---|---|
| beta | 1.0 | 0.2 | -0.06 |
| phi | 0.2 | 1.0 | -0.48 |
| drift_ratio | -0.06 | -0.48 | 1.0 |

Spearman(true parameter row, error in parameter column) - off-diagonal = leakage:

| index | beta | phi | drift_ratio |
|---|---|---|---|
| beta | 0.04 | -0.3 | 0.21 |
| phi | 0.05 | 0.16 | -0.06 |
| drift_ratio | 0.11 | -0.04 | -0.24 |

## Recovery metrics

| condition | model | param | n_trials | r | rho | median_ae | rmse | bias | frac_at_bound |
|---|---|---|---|---|---|---|---|---|---|
| trials | q | alpha | 75 | 0.454 | 0.431 | 0.192 | 0.326 | 0.016 | 0.25 |
| trials | q | beta | 75 | 0.3 | 0.662 | 0.888 | 4.462 | 1.306 | 0.05 |
| trials | q | alpha | 150 | 0.382 | 0.395 | 0.126 | 0.336 | -0.055 | 0.15 |
| trials | q | beta | 150 | 0.428 | 0.761 | 0.492 | 3.094 | 0.608 | 0.05 |
| trials | q | alpha | 300 | 0.634 | 0.591 | 0.117 | 0.261 | 0.051 | 0.275 |
| trials | q | beta | 300 | 0.931 | 0.927 | 0.394 | 0.798 | -0.099 | 0.025 |
| trials | asym_q | alpha_pos | 75 | 0.346 | 0.299 | 0.216 | 0.378 | -0.045 | 0.425 |
| trials | asym_q | alpha_neg | 75 | 0.098 | 0.092 | 0.275 | 0.41 | -0.029 | 0.225 |
| trials | asym_q | beta | 75 | 0.006 | 0.398 | 1.186 | 5.476 | 1.776 | 0.075 |
| trials | asym_q | alpha_pos | 150 | 0.558 | 0.602 | 0.158 | 0.28 | -0.048 | 0.225 |
| trials | asym_q | alpha_neg | 150 | 0.231 | 0.246 | 0.217 | 0.366 | -0.025 | 0.175 |
| trials | asym_q | beta | 150 | 0.172 | 0.64 | 0.484 | 4.178 | 1.052 | 0.025 |
| trials | asym_q | alpha_pos | 300 | 0.674 | 0.654 | 0.112 | 0.249 | 0.043 | 0.225 |
| trials | asym_q | alpha_neg | 300 | 0.529 | 0.528 | 0.133 | 0.284 | -0.029 | 0.125 |
| trials | asym_q | beta | 300 | 0.928 | 0.929 | 0.414 | 0.841 | 0.156 | 0.0 |
| trials | sticky_q | alpha | 75 | 0.562 | 0.571 | 0.154 | 0.289 | -0.026 | 0.175 |
| trials | sticky_q | beta | 75 | 0.218 | 0.489 | 0.979 | 5.169 | 1.904 | 0.075 |
| trials | sticky_q | kappa | 75 | 0.83 | 0.825 | 0.187 | 0.383 | -0.108 | 0.0 |
| trials | sticky_q | alpha | 150 | 0.682 | 0.712 | 0.104 | 0.232 | -0.042 | 0.1 |
| trials | sticky_q | beta | 150 | 0.162 | 0.678 | 0.549 | 4.881 | 1.398 | 0.075 |
| trials | sticky_q | kappa | 150 | 0.942 | 0.936 | 0.146 | 0.21 | -0.011 | 0.0 |
| trials | sticky_q | alpha | 300 | 0.747 | 0.758 | 0.097 | 0.204 | -0.001 | 0.05 |
| trials | sticky_q | beta | 300 | 0.876 | 0.907 | 0.436 | 1.044 | 0.162 | 0.0 |
| trials | sticky_q | kappa | 300 | 0.967 | 0.963 | 0.112 | 0.163 | 0.004 | 0.0 |
| trials | lapse_q | alpha | 75 | 0.473 | 0.468 | 0.174 | 0.335 | 0.037 | 0.3 |
| trials | lapse_q | beta | 75 | 0.391 | 0.576 | 1.376 | 5.205 | 1.608 | 0.075 |
| trials | lapse_q | epsilon | 75 | 0.087 | 0.026 | 0.159 | 0.208 | -0.033 | 0.7 |
| trials | lapse_q | alpha | 150 | 0.755 | 0.78 | 0.101 | 0.212 | 0.023 | 0.175 |
| trials | lapse_q | beta | 150 | 0.427 | 0.72 | 0.864 | 3.367 | 0.788 | 0.025 |
| trials | lapse_q | epsilon | 150 | 0.041 | 0.03 | 0.152 | 0.199 | -0.048 | 0.675 |
| trials | lapse_q | alpha | 300 | 0.871 | 0.872 | 0.092 | 0.154 | 0.036 | 0.175 |
| trials | lapse_q | beta | 300 | 0.81 | 0.829 | 0.808 | 1.59 | 0.301 | 0.0 |
| trials | lapse_q | epsilon | 300 | 0.376 | 0.292 | 0.151 | 0.175 | -0.015 | 0.575 |
| trials | kalman | beta | 75 | 0.839 | 0.791 | 1.232 | 2.539 | 0.894 | 0.0 |
| trials | kalman | phi | 75 | 0.505 | 0.619 | 0.4 | 1.162 | 0.49 | 0.05 |
| trials | kalman | drift_ratio | 75 | 0.149 | 0.303 | 0.604 | 3.171 | 0.902 | 0.1 |
| trials | kalman | beta | 150 | 0.952 | 0.941 | 0.783 | 1.291 | 0.505 | 0.0 |
| trials | kalman | phi | 150 | 0.684 | 0.714 | 0.329 | 0.765 | 0.319 | 0.025 |
| trials | kalman | drift_ratio | 150 | 0.295 | 0.489 | 0.535 | 3.018 | 0.778 | 0.1 |
| trials | kalman | beta | 300 | 0.985 | 0.972 | 0.474 | 0.678 | 0.115 | 0.0 |
| trials | kalman | phi | 300 | 0.737 | 0.753 | 0.274 | 0.542 | 0.2 | 0.0 |
| trials | kalman | drift_ratio | 300 | 0.374 | 0.602 | 0.321 | 2.098 | 0.364 | 0.05 |
| volatility drift_sd=0.01 | q | alpha | 150 | 0.477 | 0.49 | 0.182 | 0.308 | -0.008 | 0.225 |
| volatility drift_sd=0.01 | q | beta | 150 | 0.454 | 0.768 | 0.79 | 3.811 | 1.034 | 0.05 |
| volatility drift_sd=0.01 | asym_q | alpha_pos | 150 | 0.548 | 0.511 | 0.254 | 0.347 | -0.038 | 0.4 |
| volatility drift_sd=0.01 | asym_q | alpha_neg | 150 | -0.021 | -0.019 | 0.247 | 0.419 | -0.045 | 0.225 |
| volatility drift_sd=0.01 | asym_q | beta | 150 | 0.249 | 0.506 | 1.023 | 4.371 | 1.734 | 0.025 |
| volatility drift_sd=0.01 | sticky_q | alpha | 150 | 0.623 | 0.622 | 0.165 | 0.272 | -0.048 | 0.125 |
| volatility drift_sd=0.01 | sticky_q | beta | 150 | 0.003 | 0.491 | 0.632 | 5.143 | 1.778 | 0.075 |
| volatility drift_sd=0.01 | sticky_q | kappa | 150 | 0.951 | 0.943 | 0.121 | 0.194 | -0.048 | 0.0 |
| volatility drift_sd=0.01 | lapse_q | alpha | 150 | 0.606 | 0.57 | 0.14 | 0.274 | -0.03 | 0.175 |
| volatility drift_sd=0.01 | lapse_q | beta | 150 | 0.473 | 0.69 | 1.116 | 5.218 | 2.566 | 0.05 |
| volatility drift_sd=0.01 | lapse_q | epsilon | 150 | 0.066 | 0.058 | 0.246 | 0.252 | 0.056 | 0.825 |
| volatility drift_sd=0.01 | kalman | beta | 150 | 0.93 | 0.932 | 1.043 | 1.716 | 0.592 | 0.0 |
| volatility drift_sd=0.01 | kalman | phi | 150 | 0.574 | 0.652 | 0.385 | 1.048 | 0.45 | 0.0 |
| volatility drift_sd=0.01 | kalman | drift_ratio | 150 | 0.226 | 0.382 | 0.414 | 1.942 | 0.184 | 0.025 |
| volatility drift_sd=0.06 | q | alpha | 150 | 0.59 | 0.623 | 0.142 | 0.243 | -0.006 | 0.125 |
| volatility drift_sd=0.06 | q | beta | 150 | 0.945 | 0.949 | 0.395 | 0.748 | -0.039 | 0.0 |
| volatility drift_sd=0.06 | asym_q | alpha_pos | 150 | 0.808 | 0.829 | 0.105 | 0.202 | 0.053 | 0.25 |
| volatility drift_sd=0.06 | asym_q | alpha_neg | 150 | 0.528 | 0.541 | 0.189 | 0.272 | -0.054 | 0.125 |
| volatility drift_sd=0.06 | asym_q | beta | 150 | 0.945 | 0.931 | 0.519 | 0.717 | 0.037 | 0.025 |
| volatility drift_sd=0.06 | sticky_q | alpha | 150 | 0.739 | 0.752 | 0.108 | 0.216 | -0.016 | 0.175 |
| volatility drift_sd=0.06 | sticky_q | beta | 150 | 0.883 | 0.924 | 0.463 | 1.314 | 0.432 | 0.0 |
| volatility drift_sd=0.06 | sticky_q | kappa | 150 | 0.899 | 0.893 | 0.18 | 0.274 | -0.053 | 0.0 |
| volatility drift_sd=0.06 | lapse_q | alpha | 150 | 0.696 | 0.673 | 0.11 | 0.236 | 0.046 | 0.175 |
| volatility drift_sd=0.06 | lapse_q | beta | 150 | 0.579 | 0.718 | 0.828 | 2.687 | 0.634 | 0.0 |
| volatility drift_sd=0.06 | lapse_q | epsilon | 150 | 0.283 | 0.296 | 0.109 | 0.16 | -0.025 | 0.425 |
| volatility drift_sd=0.06 | kalman | beta | 150 | 0.924 | 0.93 | 0.614 | 1.781 | 0.7 | 0.0 |
| volatility drift_sd=0.06 | kalman | phi | 150 | 0.72 | 0.743 | 0.337 | 0.541 | 0.171 | 0.0 |
| volatility drift_sd=0.06 | kalman | drift_ratio | 150 | 0.435 | 0.553 | 0.437 | 2.834 | 0.85 | 0.1 |
| options K=2 | q | alpha | 150 | 0.346 | 0.403 | 0.196 | 0.381 | -0.044 | 0.325 |
| options K=2 | q | beta | 150 | 0.205 | 0.541 | 0.86 | 4.554 | 1.7 | 0.025 |
| options K=2 | asym_q | alpha_pos | 150 | 0.251 | 0.227 | 0.287 | 0.41 | 0.015 | 0.35 |
| options K=2 | asym_q | alpha_neg | 150 | -0.024 | -0.017 | 0.273 | 0.441 | -0.052 | 0.2 |
| options K=2 | asym_q | beta | 150 | 0.188 | 0.594 | 0.927 | 4.411 | 1.573 | 0.05 |
| options K=2 | sticky_q | alpha | 150 | 0.191 | 0.171 | 0.183 | 0.397 | -0.079 | 0.2 |
| options K=2 | sticky_q | beta | 150 | 0.443 | 0.586 | 1.108 | 3.805 | 1.352 | 0.05 |
| options K=2 | sticky_q | kappa | 150 | 0.92 | 0.918 | 0.144 | 0.237 | 0.047 | 0.0 |
| options K=2 | lapse_q | alpha | 150 | 0.231 | 0.281 | 0.298 | 0.396 | -0.099 | 0.225 |
| options K=2 | lapse_q | beta | 150 | 0.149 | 0.299 | 2.524 | 7.947 | 4.543 | 0.2 |
| options K=2 | lapse_q | epsilon | 150 | 0.045 | 0.022 | 0.213 | 0.244 | 0.066 | 0.725 |
| options K=8 | q | alpha | 150 | 0.725 | 0.732 | 0.149 | 0.227 | -0.038 | 0.175 |
| options K=8 | q | beta | 150 | 0.462 | 0.809 | 0.509 | 3.059 | 0.701 | 0.05 |
| options K=8 | asym_q | alpha_pos | 150 | 0.6 | 0.586 | 0.164 | 0.296 | 0.069 | 0.3 |
| options K=8 | asym_q | alpha_neg | 150 | 0.38 | 0.415 | 0.235 | 0.344 | 0.032 | 0.275 |
| options K=8 | asym_q | beta | 150 | 0.664 | 0.827 | 0.433 | 1.874 | 0.395 | 0.0 |
| options K=8 | sticky_q | alpha | 150 | 0.688 | 0.695 | 0.113 | 0.223 | 0.008 | 0.075 |
| options K=8 | sticky_q | beta | 150 | 0.872 | 0.872 | 0.421 | 1.147 | 0.191 | 0.0 |
| options K=8 | sticky_q | kappa | 150 | 0.948 | 0.943 | 0.135 | 0.188 | -0.019 | 0.0 |
| options K=8 | lapse_q | alpha | 150 | 0.662 | 0.679 | 0.149 | 0.246 | -0.043 | 0.1 |
| options K=8 | lapse_q | beta | 150 | 0.42 | 0.608 | 1.098 | 3.888 | 1.685 | 0.05 |
| options K=8 | lapse_q | epsilon | 150 | 0.31 | 0.297 | 0.182 | 0.219 | 0.06 | 0.625 |


## Poorly recovered (r < 0.6) - reported, not hidden

| condition | model | param | n_trials | r | rho | median_ae | rmse | bias | frac_at_bound |
|---|---|---|---|---|---|---|---|---|---|
| trials | q | alpha | 75 | 0.454 | 0.431 | 0.192 | 0.326 | 0.016 | 0.25 |
| trials | q | beta | 75 | 0.3 | 0.662 | 0.888 | 4.462 | 1.306 | 0.05 |
| trials | q | alpha | 150 | 0.382 | 0.395 | 0.126 | 0.336 | -0.055 | 0.15 |
| trials | q | beta | 150 | 0.428 | 0.761 | 0.492 | 3.094 | 0.608 | 0.05 |
| trials | asym_q | alpha_pos | 75 | 0.346 | 0.299 | 0.216 | 0.378 | -0.045 | 0.425 |
| trials | asym_q | alpha_neg | 75 | 0.098 | 0.092 | 0.275 | 0.41 | -0.029 | 0.225 |
| trials | asym_q | beta | 75 | 0.006 | 0.398 | 1.186 | 5.476 | 1.776 | 0.075 |
| trials | asym_q | alpha_pos | 150 | 0.558 | 0.602 | 0.158 | 0.28 | -0.048 | 0.225 |
| trials | asym_q | alpha_neg | 150 | 0.231 | 0.246 | 0.217 | 0.366 | -0.025 | 0.175 |
| trials | asym_q | beta | 150 | 0.172 | 0.64 | 0.484 | 4.178 | 1.052 | 0.025 |
| trials | asym_q | alpha_neg | 300 | 0.529 | 0.528 | 0.133 | 0.284 | -0.029 | 0.125 |
| trials | sticky_q | alpha | 75 | 0.562 | 0.571 | 0.154 | 0.289 | -0.026 | 0.175 |
| trials | sticky_q | beta | 75 | 0.218 | 0.489 | 0.979 | 5.169 | 1.904 | 0.075 |
| trials | sticky_q | beta | 150 | 0.162 | 0.678 | 0.549 | 4.881 | 1.398 | 0.075 |
| trials | lapse_q | alpha | 75 | 0.473 | 0.468 | 0.174 | 0.335 | 0.037 | 0.3 |
| trials | lapse_q | beta | 75 | 0.391 | 0.576 | 1.376 | 5.205 | 1.608 | 0.075 |
| trials | lapse_q | epsilon | 75 | 0.087 | 0.026 | 0.159 | 0.208 | -0.033 | 0.7 |
| trials | lapse_q | beta | 150 | 0.427 | 0.72 | 0.864 | 3.367 | 0.788 | 0.025 |
| trials | lapse_q | epsilon | 150 | 0.041 | 0.03 | 0.152 | 0.199 | -0.048 | 0.675 |
| trials | lapse_q | epsilon | 300 | 0.376 | 0.292 | 0.151 | 0.175 | -0.015 | 0.575 |
| trials | kalman | phi | 75 | 0.505 | 0.619 | 0.4 | 1.162 | 0.49 | 0.05 |
| trials | kalman | drift_ratio | 75 | 0.149 | 0.303 | 0.604 | 3.171 | 0.902 | 0.1 |
| trials | kalman | drift_ratio | 150 | 0.295 | 0.489 | 0.535 | 3.018 | 0.778 | 0.1 |
| trials | kalman | drift_ratio | 300 | 0.374 | 0.602 | 0.321 | 2.098 | 0.364 | 0.05 |
| volatility drift_sd=0.01 | q | alpha | 150 | 0.477 | 0.49 | 0.182 | 0.308 | -0.008 | 0.225 |
| volatility drift_sd=0.01 | q | beta | 150 | 0.454 | 0.768 | 0.79 | 3.811 | 1.034 | 0.05 |
| volatility drift_sd=0.01 | asym_q | alpha_pos | 150 | 0.548 | 0.511 | 0.254 | 0.347 | -0.038 | 0.4 |
| volatility drift_sd=0.01 | asym_q | alpha_neg | 150 | -0.021 | -0.019 | 0.247 | 0.419 | -0.045 | 0.225 |
| volatility drift_sd=0.01 | asym_q | beta | 150 | 0.249 | 0.506 | 1.023 | 4.371 | 1.734 | 0.025 |
| volatility drift_sd=0.01 | sticky_q | beta | 150 | 0.003 | 0.491 | 0.632 | 5.143 | 1.778 | 0.075 |
| volatility drift_sd=0.01 | lapse_q | beta | 150 | 0.473 | 0.69 | 1.116 | 5.218 | 2.566 | 0.05 |
| volatility drift_sd=0.01 | lapse_q | epsilon | 150 | 0.066 | 0.058 | 0.246 | 0.252 | 0.056 | 0.825 |
| volatility drift_sd=0.01 | kalman | phi | 150 | 0.574 | 0.652 | 0.385 | 1.048 | 0.45 | 0.0 |
| volatility drift_sd=0.01 | kalman | drift_ratio | 150 | 0.226 | 0.382 | 0.414 | 1.942 | 0.184 | 0.025 |
| volatility drift_sd=0.06 | q | alpha | 150 | 0.59 | 0.623 | 0.142 | 0.243 | -0.006 | 0.125 |
| volatility drift_sd=0.06 | asym_q | alpha_neg | 150 | 0.528 | 0.541 | 0.189 | 0.272 | -0.054 | 0.125 |
| volatility drift_sd=0.06 | lapse_q | beta | 150 | 0.579 | 0.718 | 0.828 | 2.687 | 0.634 | 0.0 |
| volatility drift_sd=0.06 | lapse_q | epsilon | 150 | 0.283 | 0.296 | 0.109 | 0.16 | -0.025 | 0.425 |
| volatility drift_sd=0.06 | kalman | drift_ratio | 150 | 0.435 | 0.553 | 0.437 | 2.834 | 0.85 | 0.1 |
| options K=2 | q | alpha | 150 | 0.346 | 0.403 | 0.196 | 0.381 | -0.044 | 0.325 |
| options K=2 | q | beta | 150 | 0.205 | 0.541 | 0.86 | 4.554 | 1.7 | 0.025 |
| options K=2 | asym_q | alpha_pos | 150 | 0.251 | 0.227 | 0.287 | 0.41 | 0.015 | 0.35 |
| options K=2 | asym_q | alpha_neg | 150 | -0.024 | -0.017 | 0.273 | 0.441 | -0.052 | 0.2 |
| options K=2 | asym_q | beta | 150 | 0.188 | 0.594 | 0.927 | 4.411 | 1.573 | 0.05 |
| options K=2 | sticky_q | alpha | 150 | 0.191 | 0.171 | 0.183 | 0.397 | -0.079 | 0.2 |
| options K=2 | sticky_q | beta | 150 | 0.443 | 0.586 | 1.108 | 3.805 | 1.352 | 0.05 |
| options K=2 | lapse_q | alpha | 150 | 0.231 | 0.281 | 0.298 | 0.396 | -0.099 | 0.225 |
| options K=2 | lapse_q | beta | 150 | 0.149 | 0.299 | 2.524 | 7.947 | 4.543 | 0.2 |
| options K=2 | lapse_q | epsilon | 150 | 0.045 | 0.022 | 0.213 | 0.244 | 0.066 | 0.725 |
| options K=8 | q | beta | 150 | 0.462 | 0.809 | 0.509 | 3.059 | 0.701 | 0.05 |
| options K=8 | asym_q | alpha_pos | 150 | 0.6 | 0.586 | 0.164 | 0.296 | 0.069 | 0.3 |
| options K=8 | asym_q | alpha_neg | 150 | 0.38 | 0.415 | 0.235 | 0.344 | 0.032 | 0.275 |
| options K=8 | lapse_q | beta | 150 | 0.42 | 0.608 | 1.098 | 3.888 | 1.685 | 0.05 |
| options K=8 | lapse_q | epsilon | 150 | 0.31 | 0.297 | 0.182 | 0.219 | 0.06 | 0.625 |
