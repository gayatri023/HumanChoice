"""K-option tutorial: a 4-armed restless bandit with continuous rewards and missed trials.

Simulate a participant with perseverative learning, fit the model ladder, check which mechanisms the data can support and
whether each model reproduces the participant's stay behaviour.

Run:  python examples/06_four_option_bandit_with_timeouts.py
All data are SYNTHETIC.
"""

import numpy as np

from humanchoice.envs import RestlessBandit
from humanchoice.evaluation.predictive_checks import one_step_ahead_check
from humanchoice.fitting import fit_model, likelihood_profile, profile_interval
from humanchoice.models import AsymmetricQLearning, KalmanBandit, QLearning, RandomChoice, StickyQLearning, WinStayLoseShift
from humanchoice.utils import MISSED

K = 4
truth = StickyQLearning(K)
theta = [0.35, 6.0, 1.2]  # alpha, beta, kappa (strong perseveration)
sim = truth.simulate(theta, RestlessBandit(K), n_trials=300, rng=11, missed_prob=0.04)  # ~4% timeouts
c, r = sim["choices"], sim["rewards"]
print(f"{(c == MISSED).sum()} of {len(c)} trials were missed (choice = -1): no outcome, no learning, no likelihood contribution")

ladder = {"random": RandomChoice(K), "wsls": WinStayLoseShift(K), "q": QLearning(K), "asym_q": AsymmetricQLearning(K),
          "sticky_q": StickyQLearning(K), "kalman": KalmanBandit(K)}
print(f"\n{'model':9s} {'k':>2s} {'NLL':>8s} {'BIC':>8s}   estimates")
fits = {}
for name, m in ladder.items():
    f = fit_model(m, c, r, n_starts=5, seed=0)
    fits[name] = f
    print(f"{name:9s} {f.n_params:2d} {f.nll:8.1f} {f.bic:8.1f}   " + ", ".join(f"{k}={v:.2f}" for k, v in f.params.items()))
print("generating values for sticky_q: alpha=0.35, beta=6.0, kappa=1.2")

# how well does each model reproduce stay behaviour? (one-step-ahead predictive check; p < .05 = systematic misfit)
print("\nstay-rate predictive check (observed vs model replicates):")
for name in ("q", "sticky_q"):
    chk = one_step_ahead_check(ladder[name], list(fits[name].params.values()), c, r, n_sims=300, rng=0).set_index("statistic")
    row = chk.loc["p_stay"]
    print(f"  {name:9s} observed {row.observed:.2f}  replicates {row.replicate_mean:.2f} [{row.lo:.2f}, {row.hi:.2f}]  p = {row.p_value:.3f}")

# is kappa identified from this participant's data? likelihood profile
prof = likelihood_profile(ladder["sticky_q"], c, r, "kappa", n_grid=21, n_starts=2)
lo, hi, is_open = profile_interval(prof)
print(f"\n95% likelihood interval for kappa: [{lo:.2f}, {hi:.2f}] ({'open - not identified' if is_open else 'closed'}); true value 1.2")
assert np.isfinite(lo)
