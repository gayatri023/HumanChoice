# Experiment 4 - parameter recovery summary (SYNTHETIC data)

r = Pearson correlation (95% bootstrap CI) between generating and recovered parameters; rho = Spearman rank correlation; median_ae = median absolute error. Beta estimates have heavy-tailed errors (a few fits run to the upper bound), which depress r and inflate rmse/bias much more than rho and median_ae - compare them.

| model | param | n_trials | r | r_ci_lo | r_ci_hi | rho | median_ae | rmse | bias | frac_at_bound |
|---|---|---|---|---|---|---|---|---|---|---|
| q_learning | alpha | 50 | 0.731 | 0.577 | 0.848 | 0.745 | 0.122 | 0.225 | -0.044 | 0.1 |
| q_learning | beta | 50 | 0.609 | 0.39 | 0.78 | 0.729 | 1.44 | 6.377 | 3.273 | 0.167 |
| q_learning | alpha | 100 | 0.853 | 0.739 | 0.934 | 0.845 | 0.073 | 0.16 | -0.019 | 0.067 |
| q_learning | beta | 100 | 0.601 | 0.434 | 0.769 | 0.832 | 0.672 | 5.543 | 2.386 | 0.183 |
| q_learning | alpha | 200 | 0.829 | 0.685 | 0.939 | 0.852 | 0.055 | 0.167 | -0.027 | 0.083 |
| q_learning | beta | 200 | 0.609 | 0.334 | 0.908 | 0.876 | 0.54 | 3.756 | 1.057 | 0.117 |
| q_learning | alpha | 400 | 0.876 | 0.739 | 0.966 | 0.907 | 0.034 | 0.136 | -0.025 | 0.067 |
| q_learning | beta | 400 | 0.925 | 0.879 | 0.964 | 0.957 | 0.432 | 1.066 | 0.186 | 0.05 |
| asymmetric_q_learning | alpha_pos | 50 | 0.562 | 0.348 | 0.718 | 0.541 | 0.182 | 0.322 | -0.036 | 0.283 |
| asymmetric_q_learning | alpha_neg | 50 | 0.709 | 0.529 | 0.831 | 0.681 | 0.147 | 0.225 | 0.022 | 0.15 |
| asymmetric_q_learning | beta | 50 | 0.325 | 0.071 | 0.567 | 0.443 | 2.315 | 9.188 | 5.868 | 0.3 |
| asymmetric_q_learning | alpha_pos | 100 | 0.689 | 0.497 | 0.823 | 0.671 | 0.14 | 0.246 | 0.008 | 0.217 |
| asymmetric_q_learning | alpha_neg | 100 | 0.804 | 0.658 | 0.895 | 0.775 | 0.128 | 0.19 | -0.013 | 0.133 |
| asymmetric_q_learning | beta | 100 | 0.463 | 0.188 | 0.691 | 0.654 | 1.086 | 6.494 | 2.878 | 0.2 |
| asymmetric_q_learning | alpha_pos | 200 | 0.809 | 0.674 | 0.901 | 0.817 | 0.104 | 0.186 | -0.018 | 0.133 |
| asymmetric_q_learning | alpha_neg | 200 | 0.849 | 0.712 | 0.93 | 0.834 | 0.099 | 0.167 | -0.017 | 0.083 |
| asymmetric_q_learning | beta | 200 | 0.593 | 0.32 | 0.769 | 0.786 | 0.815 | 5.017 | 1.97 | 0.15 |
| asymmetric_q_learning | alpha_pos | 400 | 0.921 | 0.883 | 0.948 | 0.905 | 0.07 | 0.123 | -0.014 | 0.1 |
| asymmetric_q_learning | alpha_neg | 400 | 0.879 | 0.726 | 0.961 | 0.873 | 0.055 | 0.142 | -0.009 | 0.083 |
| asymmetric_q_learning | beta | 400 | 0.649 | 0.472 | 0.867 | 0.874 | 0.683 | 3.016 | 0.864 | 0.05 |
| pomdp_reversal | beta | 50 | 0.242 | -0.043 | 0.523 | 0.43 | 4.142 | 14.452 | 7.731 | 0.167 |
| pomdp_reversal | hazard | 50 | 0.728 | 0.575 | 0.847 | 0.794 | 0.041 | 0.096 | 0.035 | 0.0 |
| pomdp_reversal | beta | 100 | 0.316 | 0.069 | 0.647 | 0.672 | 1.398 | 9.605 | 3.578 | 0.083 |
| pomdp_reversal | hazard | 100 | 0.813 | 0.724 | 0.887 | 0.875 | 0.022 | 0.07 | 0.022 | 0.0 |
| pomdp_reversal | beta | 200 | 0.448 | 0.073 | 0.902 | 0.767 | 1.177 | 5.363 | 1.006 | 0.017 |
| pomdp_reversal | hazard | 200 | 0.821 | 0.731 | 0.915 | 0.898 | 0.017 | 0.053 | -0.001 | 0.0 |
| pomdp_reversal | beta | 400 | 0.913 | 0.86 | 0.949 | 0.909 | 0.849 | 1.58 | 0.132 | 0.0 |
| pomdp_reversal | hazard | 400 | 0.944 | 0.909 | 0.971 | 0.956 | 0.012 | 0.027 | -0.003 | 0.0 |
| bamdp_bandit | beta | 50 | 0.825 | 0.751 | 0.893 | 0.845 | 1.323 | 2.063 | -0.353 | 0.017 |
| bamdp_bandit | beta | 100 | 0.896 | 0.844 | 0.944 | 0.916 | 0.683 | 1.594 | -0.17 | 0.0 |
| bamdp_bandit | beta | 200 | 0.936 | 0.887 | 0.967 | 0.933 | 0.737 | 1.195 | -0.301 | 0.017 |
| bamdp_bandit | beta | 400 | 0.95 | 0.906 | 0.975 | 0.941 | 0.469 | 0.988 | -0.19 | 0.0 |



## Poorly recovered (r < 0.6) - reported, not hidden

| model | param | n_trials | r | r_ci_lo | r_ci_hi | rho | median_ae | rmse | bias | frac_at_bound |
|---|---|---|---|---|---|---|---|---|---|---|
| asymmetric_q_learning | alpha_pos | 50 | 0.562 | 0.348 | 0.718 | 0.541 | 0.182 | 0.322 | -0.036 | 0.283 |
| asymmetric_q_learning | beta | 50 | 0.325 | 0.071 | 0.567 | 0.443 | 2.315 | 9.188 | 5.868 | 0.3 |
| asymmetric_q_learning | beta | 100 | 0.463 | 0.188 | 0.691 | 0.654 | 1.086 | 6.494 | 2.878 | 0.2 |
| asymmetric_q_learning | beta | 200 | 0.593 | 0.32 | 0.769 | 0.786 | 0.815 | 5.017 | 1.97 | 0.15 |
| pomdp_reversal | beta | 50 | 0.242 | -0.043 | 0.523 | 0.43 | 4.142 | 14.452 | 7.731 | 0.167 |
| pomdp_reversal | beta | 100 | 0.316 | 0.069 | 0.647 | 0.672 | 1.398 | 9.605 | 3.578 | 0.083 |
| pomdp_reversal | beta | 200 | 0.448 | 0.073 | 0.902 | 0.767 | 1.177 | 5.363 | 1.006 | 0.017 |

## Recovery by noise level (inverse-temperature tercile)

| model | n_trials | stratum | by_min | by_max | param | r | rho | median_ae | rmse | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| q_learning | 50 | Q1 (lowest) | 0.653 | 2.86 | alpha | 0.474 | 0.464 | 0.247 | 0.33 | -0.108 |
| q_learning | 50 | Q1 (lowest) | 0.653 | 2.86 | beta | -0.212 | 0.092 | 0.838 | 4.385 | 1.144 |
| q_learning | 50 | Q2 | 2.986 | 5.159 | alpha | 0.872 | 0.868 | 0.077 | 0.155 | -0.06 |
| q_learning | 50 | Q2 | 2.986 | 5.159 | beta | 0.432 | 0.572 | 1.246 | 6.785 | 3.566 |
| q_learning | 50 | Q3 (highest) | 5.451 | 7.882 | alpha | 0.912 | 0.904 | 0.091 | 0.134 | 0.035 |
| q_learning | 50 | Q3 (highest) | 5.451 | 7.882 | beta | 0.44 | 0.405 | 3.923 | 7.532 | 5.108 |
| q_learning | 100 | Q1 (lowest) | 0.653 | 2.86 | alpha | 0.704 | 0.669 | 0.125 | 0.25 | -0.072 |
| q_learning | 100 | Q1 (lowest) | 0.653 | 2.86 | beta | 0.518 | 0.685 | 0.52 | 3.902 | 0.78 |
| q_learning | 100 | Q2 | 2.986 | 5.159 | alpha | 0.961 | 0.967 | 0.055 | 0.079 | -0.021 |
| q_learning | 100 | Q2 | 2.986 | 5.159 | beta | 0.448 | 0.752 | 0.602 | 6.214 | 2.864 |
| q_learning | 100 | Q3 (highest) | 5.451 | 7.882 | alpha | 0.956 | 0.929 | 0.055 | 0.091 | 0.035 |
| q_learning | 100 | Q3 (highest) | 5.451 | 7.882 | beta | 0.246 | 0.302 | 2.539 | 6.191 | 3.515 |
| q_learning | 200 | Q1 (lowest) | 0.653 | 2.86 | alpha | 0.645 | 0.648 | 0.107 | 0.272 | -0.095 |
| q_learning | 200 | Q1 (lowest) | 0.653 | 2.86 | beta | 0.098 | 0.771 | 0.427 | 4.16 | 0.665 |
| q_learning | 200 | Q2 | 2.986 | 5.159 | alpha | 0.972 | 0.985 | 0.04 | 0.067 | -0.025 |
| q_learning | 200 | Q2 | 2.986 | 5.159 | beta | 0.725 | 0.723 | 0.481 | 0.592 | 0.141 |
| q_learning | 200 | Q3 (highest) | 5.451 | 7.882 | alpha | 0.976 | 0.947 | 0.039 | 0.073 | 0.039 |
| q_learning | 200 | Q3 (highest) | 5.451 | 7.882 | beta | 0.083 | 0.378 | 1.509 | 4.967 | 2.366 |
| q_learning | 400 | Q1 (lowest) | 0.653 | 2.86 | alpha | 0.732 | 0.789 | 0.069 | 0.224 | -0.096 |
| q_learning | 400 | Q1 (lowest) | 0.653 | 2.86 | beta | 0.778 | 0.752 | 0.222 | 0.57 | -0.126 |
| q_learning | 400 | Q2 | 2.986 | 5.159 | alpha | 0.981 | 0.982 | 0.034 | 0.051 | -0.005 |
| q_learning | 400 | Q2 | 2.986 | 5.159 | beta | 0.813 | 0.823 | 0.42 | 0.502 | -0.023 |
| q_learning | 400 | Q3 (highest) | 5.451 | 7.882 | alpha | 0.983 | 0.968 | 0.012 | 0.054 | 0.026 |
| q_learning | 400 | Q3 (highest) | 5.451 | 7.882 | beta | 0.348 | 0.411 | 0.727 | 1.684 | 0.707 |
| asymmetric_q_learning | 50 | Q3 (highest) | 6.06 | 7.882 | alpha_pos | 0.712 | 0.744 | 0.182 | 0.273 | -0.021 |
| asymmetric_q_learning | 50 | Q3 (highest) | 6.06 | 7.882 | alpha_neg | 0.557 | 0.517 | 0.136 | 0.241 | 0.043 |
| asymmetric_q_learning | 50 | Q3 (highest) | 6.06 | 7.882 | beta | 0.317 | 0.301 | 6.383 | 8.318 | 5.883 |
| asymmetric_q_learning | 50 | Q1 (lowest) | 0.533 | 3.224 | alpha_pos | 0.273 | 0.216 | 0.249 | 0.369 | -0.019 |
| asymmetric_q_learning | 50 | Q1 (lowest) | 0.533 | 3.224 | alpha_neg | 0.786 | 0.795 | 0.157 | 0.22 | 0.013 |
| asymmetric_q_learning | 50 | Q1 (lowest) | 0.533 | 3.224 | beta | -0.521 | -0.224 | 0.941 | 8.363 | 4.202 |
| asymmetric_q_learning | 50 | Q2 | 3.238 | 5.467 | alpha_pos | 0.621 | 0.67 | 0.107 | 0.317 | -0.068 |
| asymmetric_q_learning | 50 | Q2 | 3.238 | 5.467 | alpha_neg | 0.704 | 0.752 | 0.133 | 0.214 | 0.012 |
| asymmetric_q_learning | 50 | Q2 | 3.238 | 5.467 | beta | 0.08 | 0.197 | 6.205 | 10.684 | 7.52 |
| asymmetric_q_learning | 100 | Q3 (highest) | 6.06 | 7.882 | alpha_pos | 0.84 | 0.867 | 0.118 | 0.169 | -0.005 |
| asymmetric_q_learning | 100 | Q3 (highest) | 6.06 | 7.882 | alpha_neg | 0.873 | 0.811 | 0.091 | 0.13 | -0.018 |
| asymmetric_q_learning | 100 | Q3 (highest) | 6.06 | 7.882 | beta | 0.291 | 0.318 | 2.451 | 7.812 | 4.562 |
| asymmetric_q_learning | 100 | Q1 (lowest) | 0.533 | 3.224 | alpha_pos | 0.575 | 0.549 | 0.152 | 0.299 | -0.029 |
| asymmetric_q_learning | 100 | Q1 (lowest) | 0.533 | 3.224 | alpha_neg | 0.73 | 0.69 | 0.178 | 0.268 | -0.011 |
| asymmetric_q_learning | 100 | Q1 (lowest) | 0.533 | 3.224 | beta | -0.337 | 0.081 | 1.005 | 6.121 | 2.366 |
| asymmetric_q_learning | 100 | Q2 | 3.238 | 5.467 | alpha_pos | 0.677 | 0.744 | 0.164 | 0.25 | 0.059 |
| asymmetric_q_learning | 100 | Q2 | 3.238 | 5.467 | alpha_neg | 0.849 | 0.846 | 0.121 | 0.14 | -0.009 |
| asymmetric_q_learning | 100 | Q2 | 3.238 | 5.467 | beta | 0.328 | 0.353 | 0.846 | 5.294 | 1.707 |
| asymmetric_q_learning | 200 | Q3 (highest) | 6.06 | 7.882 | alpha_pos | 0.814 | 0.822 | 0.085 | 0.179 | -0.066 |
| asymmetric_q_learning | 200 | Q3 (highest) | 6.06 | 7.882 | alpha_neg | 0.902 | 0.886 | 0.099 | 0.116 | 0.007 |
| asymmetric_q_learning | 200 | Q3 (highest) | 6.06 | 7.882 | beta | 0.25 | 0.29 | 1.846 | 7.284 | 3.976 |
| asymmetric_q_learning | 200 | Q1 (lowest) | 0.533 | 3.224 | alpha_pos | 0.651 | 0.726 | 0.152 | 0.239 | -0.032 |
| asymmetric_q_learning | 200 | Q1 (lowest) | 0.533 | 3.224 | alpha_neg | 0.795 | 0.672 | 0.106 | 0.233 | -0.041 |
| asymmetric_q_learning | 200 | Q1 (lowest) | 0.533 | 3.224 | beta | -0.058 | 0.515 | 0.405 | 4.269 | 1.165 |
| asymmetric_q_learning | 200 | Q2 | 3.238 | 5.467 | alpha_pos | 0.946 | 0.889 | 0.072 | 0.121 | 0.043 |
| asymmetric_q_learning | 200 | Q2 | 3.238 | 5.467 | alpha_neg | 0.876 | 0.876 | 0.078 | 0.126 | -0.016 |
| asymmetric_q_learning | 200 | Q2 | 3.238 | 5.467 | beta | 0.425 | 0.504 | 0.896 | 2.061 | 0.769 |
| asymmetric_q_learning | 400 | Q3 (highest) | 6.06 | 7.882 | alpha_pos | 0.939 | 0.931 | 0.05 | 0.1 | -0.021 |
| asymmetric_q_learning | 400 | Q3 (highest) | 6.06 | 7.882 | alpha_neg | 0.968 | 0.941 | 0.043 | 0.064 | -0.011 |
| asymmetric_q_learning | 400 | Q3 (highest) | 6.06 | 7.882 | beta | 0.543 | 0.602 | 0.878 | 2.568 | 0.617 |
| asymmetric_q_learning | 400 | Q1 (lowest) | 0.533 | 3.224 | alpha_pos | 0.877 | 0.908 | 0.115 | 0.147 | -0.049 |
| asymmetric_q_learning | 400 | Q1 (lowest) | 0.533 | 3.224 | alpha_neg | 0.779 | 0.745 | 0.089 | 0.219 | -0.029 |
| asymmetric_q_learning | 400 | Q1 (lowest) | 0.533 | 3.224 | beta | 0.641 | 0.658 | 0.353 | 0.872 | 0.316 |
| asymmetric_q_learning | 400 | Q2 | 3.238 | 5.467 | alpha_pos | 0.945 | 0.91 | 0.076 | 0.116 | 0.029 |
| asymmetric_q_learning | 400 | Q2 | 3.238 | 5.467 | alpha_neg | 0.941 | 0.929 | 0.051 | 0.091 | 0.013 |
| asymmetric_q_learning | 400 | Q2 | 3.238 | 5.467 | beta | 0.342 | 0.648 | 0.603 | 4.465 | 1.659 |
| pomdp_reversal | 50 | Q1 (lowest) | 2.579 | 7.217 | beta | 0.059 | 0.174 | 1.451 | 19.283 | 11.26 |
| pomdp_reversal | 50 | Q1 (lowest) | 2.579 | 7.217 | hazard | 0.859 | 0.86 | 0.037 | 0.131 | 0.074 |
| pomdp_reversal | 50 | Q3 (highest) | 11.883 | 14.971 | beta | 0.102 | 0.19 | 5.797 | 13.947 | 9.386 |
| pomdp_reversal | 50 | Q3 (highest) | 11.883 | 14.971 | hazard | 0.832 | 0.851 | 0.027 | 0.065 | 0.022 |
| pomdp_reversal | 50 | Q2 | 7.298 | 10.897 | beta | 0.335 | 0.248 | 3.212 | 7.759 | 2.548 |
| pomdp_reversal | 50 | Q2 | 7.298 | 10.897 | hazard | 0.598 | 0.62 | 0.058 | 0.081 | 0.008 |
| pomdp_reversal | 100 | Q1 (lowest) | 2.579 | 7.217 | beta | 0.16 | 0.377 | 1.241 | 11.355 | 4.431 |
| pomdp_reversal | 100 | Q1 (lowest) | 2.579 | 7.217 | hazard | 0.776 | 0.803 | 0.037 | 0.098 | 0.038 |
| pomdp_reversal | 100 | Q3 (highest) | 11.883 | 14.971 | beta | 0.244 | 0.274 | 1.73 | 6.759 | 2.688 |
| pomdp_reversal | 100 | Q3 (highest) | 11.883 | 14.971 | hazard | 0.921 | 0.877 | 0.009 | 0.031 | 0.006 |
| pomdp_reversal | 100 | Q2 | 7.298 | 10.897 | beta | -0.021 | 0.107 | 1.318 | 10.108 | 3.614 |
| pomdp_reversal | 100 | Q2 | 7.298 | 10.897 | hazard | 0.853 | 0.91 | 0.022 | 0.063 | 0.021 |
| pomdp_reversal | 200 | Q1 (lowest) | 2.579 | 7.217 | beta | 0.008 | 0.68 | 0.558 | 8.281 | 2.04 |
| pomdp_reversal | 200 | Q1 (lowest) | 2.579 | 7.217 | hazard | 0.734 | 0.901 | 0.027 | 0.072 | 0.003 |
| pomdp_reversal | 200 | Q3 (highest) | 11.883 | 14.971 | beta | 0.403 | 0.382 | 1.068 | 1.537 | 0.01 |
| pomdp_reversal | 200 | Q3 (highest) | 11.883 | 14.971 | hazard | 0.928 | 0.931 | 0.013 | 0.029 | -0.003 |
| pomdp_reversal | 200 | Q2 | 7.298 | 10.897 | beta | 0.081 | 0.03 | 1.805 | 3.915 | 0.967 |
| pomdp_reversal | 200 | Q2 | 7.298 | 10.897 | hazard | 0.854 | 0.916 | 0.019 | 0.047 | -0.003 |
| pomdp_reversal | 400 | Q1 (lowest) | 2.579 | 7.217 | beta | 0.868 | 0.925 | 0.412 | 1.009 | 0.169 |
| pomdp_reversal | 400 | Q1 (lowest) | 2.579 | 7.217 | hazard | 0.938 | 0.944 | 0.01 | 0.03 | -0.0 |
| pomdp_reversal | 400 | Q3 (highest) | 11.883 | 14.971 | beta | 0.191 | 0.066 | 1.13 | 1.996 | -0.167 |
| pomdp_reversal | 400 | Q3 (highest) | 11.883 | 14.971 | hazard | 0.974 | 0.967 | 0.01 | 0.021 | -0.01 |
| pomdp_reversal | 400 | Q2 | 7.298 | 10.897 | beta | 0.424 | 0.489 | 1.316 | 1.577 | 0.394 |
| pomdp_reversal | 400 | Q2 | 7.298 | 10.897 | hazard | 0.941 | 0.946 | 0.016 | 0.029 | 0.002 |
| bamdp_bandit | 50 | Q1 (lowest) | 1.225 | 5.272 | beta | 0.506 | 0.377 | 1.174 | 1.712 | -0.319 |
| bamdp_bandit | 50 | Q3 (highest) | 8.28 | 11.975 | beta | 0.499 | 0.592 | 1.341 | 2.499 | -0.457 |
| bamdp_bandit | 50 | Q2 | 5.283 | 8.261 | beta | 0.083 | 0.242 | 1.387 | 1.895 | -0.283 |
| bamdp_bandit | 100 | Q1 (lowest) | 1.225 | 5.272 | beta | 0.66 | 0.681 | 0.674 | 1.264 | -0.475 |
| bamdp_bandit | 100 | Q3 (highest) | 8.28 | 11.975 | beta | 0.514 | 0.663 | 0.653 | 2.083 | -0.054 |
| bamdp_bandit | 100 | Q2 | 5.283 | 8.261 | beta | 0.32 | 0.376 | 0.819 | 1.298 | 0.018 |
| bamdp_bandit | 200 | Q1 (lowest) | 1.225 | 5.272 | beta | 0.672 | 0.618 | 0.783 | 1.192 | -0.521 |
| bamdp_bandit | 200 | Q3 (highest) | 8.28 | 11.975 | beta | 0.685 | 0.758 | 0.63 | 1.249 | -0.342 |
| bamdp_bandit | 200 | Q2 | 5.283 | 8.261 | beta | 0.384 | 0.398 | 0.908 | 1.141 | -0.042 |
| bamdp_bandit | 400 | Q1 (lowest) | 1.225 | 5.272 | beta | 0.81 | 0.735 | 0.438 | 0.781 | -0.092 |
| bamdp_bandit | 400 | Q3 (highest) | 8.28 | 11.975 | beta | 0.636 | 0.68 | 0.514 | 1.211 | -0.447 |
| bamdp_bandit | 400 | Q2 | 5.283 | 8.261 | beta | 0.378 | 0.465 | 0.522 | 0.924 | -0.031 |
