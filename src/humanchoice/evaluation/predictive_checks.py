"""Simulation-based model checking: does the fitted model reproduce the *features* of the data?

Two kinds of replicate are provided; neither is a substitute for held-out prediction.

* :func:`one_step_ahead_check` - replicate each choice from the model's one-step-ahead predictive distribution *given the
  participant's actual history*; "stay" statistics compare each replicate with the *observed* previous choice. Works for any dataset (no environment needed) and isolates what the model gets wrong
  about each decision.
* :func:`free_running_check` - let the fitted model interact with a simulated environment for whole sessions.
  Needs the task's reward process; it tests whether the model reproduces behaviour when its own choices drive learning.

A statistic whose observed value falls far in the tail of the replicate distribution (small ``p_value``) is a
*systematic* misfit. Across a population, roughly 5 % of participants are expected at p < 0.05 by chance alone; a model is
**flagged** on a statistic only if the misfit rate exceeds the pre-declared threshold (:func:`misfit_rate`).

Statistics come in two families. *Choice-only* statistics (stay rates, entropy, option concentration) are evaluated by the
one-step-ahead check. *Outcome* statistics (mean / total reward, reward spread, improvement over the session) need simulated
rewards and are therefore evaluated **only** by the free-running check, in an environment that must be justified for the task.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from humanchoice.models.base import ChoiceModel
from humanchoice.utils import MISSED, as_rng

#: a statistic maps a *trial table* (see :func:`_table`) and the number of options to a number
Statistic = Callable[[dict, int], float]


def _table(choices: NDArray, rewards: NDArray | None, win_threshold: float) -> dict:
    """Trial table: current choice, previous *observed* choice, whether the previous observed trial was a win, reward.

    ``rewards=None`` is used for one-step-ahead replicates, whose outcomes are unknown (reward statistics -> NaN).
    """
    c = np.asarray(choices, dtype=int)
    prev = np.full(len(c), MISSED, dtype=int)
    prev_win = np.zeros(len(c), dtype=bool)
    last, last_win = MISSED, False
    for t in range(len(c)):
        prev[t], prev_win[t] = last, last_win
        if c[t] != MISSED:
            last = c[t]
            if rewards is not None:
                last_win = bool(rewards[t] >= win_threshold)
    return {"cur": c, "prev": prev, "prev_win": prev_win, "reward": None if rewards is None else np.asarray(rewards, float)}


def _with_history(replicate: NDArray, observed_table: dict) -> dict:
    """A one-step-ahead replicate: new current choices, but the *observed* history as the conditioning information."""
    return {**observed_table, "cur": replicate, "reward": None}


def make_statistics(win_threshold: float = 0.5) -> dict[str, Statistic]:
    """Default behavioural summary statistics (computed on observed trials only).

    A *stay* means the current choice equals the previous observed choice; ``win`` means the previous observed reward
    was ``>= win_threshold``. For continuous rewards pick the threshold deliberately and report it. ``mean_reward`` is
    only defined when outcomes exist (observed data and free-running replicates), not for one-step-ahead replicates.
    """

    def stays(tab):
        valid = (tab["cur"] != MISSED) & (tab["prev"] != MISSED)
        return (tab["cur"] == tab["prev"])[valid].astype(float), tab["prev_win"][valid]

    def mean_or_nan(x):
        return float(np.mean(x)) if len(x) else np.nan

    def p_stay(tab, k):
        s, _ = stays(tab)
        return mean_or_nan(s)

    def p_stay_win(tab, k):
        s, w = stays(tab)
        return mean_or_nan(s[w])

    def p_stay_loss(tab, k):
        s, w = stays(tab)
        return mean_or_nan(s[~w])

    def stay_early(tab, k):
        s, _ = stays(tab)
        return mean_or_nan(s[: max(1, len(s) // 3)]) if len(s) else np.nan

    def stay_late(tab, k):
        s, _ = stays(tab)
        return mean_or_nan(s[-max(1, len(s) // 3):]) if len(s) else np.nan

    def choice_entropy(tab, k):
        cc = tab["cur"][tab["cur"] != MISSED]
        if not len(cc):
            return np.nan
        f = np.bincount(cc, minlength=k) / len(cc)
        f = f[f > 0]
        return float(-(f * np.log(f)).sum())

    def max_option_share(tab, k):
        cc = tab["cur"][tab["cur"] != MISSED]
        return float(np.bincount(cc, minlength=k).max() / len(cc)) if len(cc) else np.nan

    def _rewards(tab):
        if tab["reward"] is None:
            raise ValueError("outcome statistics need simulated rewards: use free_running_check, not one_step_ahead_check")
        return tab["reward"][tab["cur"] != MISSED]

    def mean_reward(tab, k):
        r = _rewards(tab)
        return mean_or_nan(r)

    def total_reward(tab, k):
        r = _rewards(tab)
        return float(r.sum()) if len(r) else np.nan

    def reward_sd(tab, k):
        r = _rewards(tab)
        return float(r.std()) if len(r) > 1 else np.nan

    def reward_gain(tab, k):  # mean reward in the last third minus the first third: did behaviour improve over the session?
        r = _rewards(tab)
        n = max(1, len(r) // 3)
        return float(r[-n:].mean() - r[:n].mean()) if len(r) >= 3 else np.nan

    return {
        "p_stay": p_stay,
        "p_stay_after_win": p_stay_win,
        "p_stay_after_loss": p_stay_loss,
        "p_stay_first_third": stay_early,
        "p_stay_last_third": stay_late,
        "choice_entropy": choice_entropy,
        "max_option_share": max_option_share,
        "mean_reward": mean_reward,
        "total_reward": total_reward,
        "reward_sd": reward_sd,
        "reward_gain_last_minus_first_third": reward_gain,
    }


#: statistics computed from choices alone (valid for one-step-ahead replicates)
ONE_STEP_STATISTICS = (
    "p_stay", "p_stay_after_win", "p_stay_after_loss", "p_stay_first_third", "p_stay_last_third", "choice_entropy", "max_option_share",
)
#: statistics that need the task's outcomes (valid only for free-running replicates in a justified environment)
SESSION_STATISTICS = ("mean_reward", "total_reward", "reward_sd", "reward_gain_last_minus_first_third")


def make_one_step_statistics(win_threshold: float = 0.5) -> dict[str, Statistic]:
    """The seven choice-only statistics evaluated by :func:`one_step_ahead_check`."""
    allst = make_statistics(win_threshold)
    return {k: allst[k] for k in ONE_STEP_STATISTICS}


def make_session_statistics(win_threshold: float = 0.5) -> dict[str, Statistic]:
    """Outcome statistics for :func:`free_running_check` (mean / total / spread of reward and improvement over the session)."""
    allst = make_statistics(win_threshold)
    return {k: allst[k] for k in SESSION_STATISTICS}


def _summarise(observed: dict[str, float], replicates: dict[str, NDArray], level: float = 0.95) -> pd.DataFrame:
    rows = []
    a = (1 - level) / 2 * 100
    for name, obs in observed.items():
        rep = replicates[name]
        rep = rep[np.isfinite(rep)]
        if not len(rep) or not np.isfinite(obs):
            rows.append({"statistic": name, "observed": obs, "replicate_mean": np.nan, "lo": np.nan, "hi": np.nan, "p_value": np.nan})
            continue
        k_hi, k_lo = int((rep >= obs).sum()), int((rep <= obs).sum())
        rows.append(
            {
                "statistic": name,
                "observed": float(obs),
                "replicate_mean": float(rep.mean()),
                "lo": float(np.percentile(rep, a)),
                "hi": float(np.percentile(rep, 100 - a)),
                # finite-simulation correction: with n replicates the smallest attainable two-sided p is 2/(n+1), never exactly 0
                "p_value": float(min(1.0, 2 * (min(k_hi, k_lo) + 1) / (len(rep) + 1))),
            }
        )
    return pd.DataFrame(rows)


def one_step_ahead_check(
    model: ChoiceModel,
    params: ArrayLike,
    choices: ArrayLike,
    rewards: ArrayLike,
    available: ArrayLike | None = None,
    n_sims: int = 500,
    statistics: dict[str, Statistic] | None = None,
    rng: int | np.random.Generator | None = 0,
    win_threshold: float = 0.5,
) -> pd.DataFrame:
    """Replicate choices from the one-step-ahead predictive distribution given the observed history.

    Returns one row per statistic: observed value, replicate mean and 95 % interval, and a two-sided tail ``p_value``.
    Because every replicate shares the participant's real history, this checks the decision rule and the value
    updating, not the model's behaviour in a free-running world.
    """
    rng = as_rng(rng)
    c = np.asarray(choices, dtype=int)
    r = np.asarray(rewards, dtype=float)
    stats = statistics or make_one_step_statistics(win_threshold)
    traj = model.trajectory(params, c, r, available)
    probs, obs = traj["probs"], traj["observed"]
    cum = np.cumsum(probs, axis=1)
    u = rng.random((n_sims, len(c), 1))
    sims = np.minimum((u > cum[None]).sum(axis=2), model.n_actions - 1)
    sims[:, ~obs] = MISSED
    obs_tab = _table(c, r, win_threshold)
    observed = {k: f(obs_tab, model.n_actions) for k, f in stats.items()}
    reps_tabs = [_with_history(sims[i], obs_tab) for i in range(n_sims)]  # reward=None -> outcome statistics raise (never silently NaN)
    replicates = {k: np.array([f(t, model.n_actions) for t in reps_tabs]) for k, f in stats.items()}
    return _summarise(observed, replicates)


def free_running_check(
    model: ChoiceModel,
    params: ArrayLike,
    env_factory: Callable[[], object],
    choices: ArrayLike,
    rewards: ArrayLike,
    n_sims: int = 200,
    statistics: dict[str, Statistic] | None = None,
    rng: int | np.random.Generator | None = 0,
    win_threshold: float = 0.5,
    available: ArrayLike | None = None,
    missed: str | float = "observed",
) -> pd.DataFrame:
    """Simulate ``n_sims`` full sessions of the fitted model in the task and compare their statistics with the data's.

    Simulated sessions must be comparable with the observed one, so missingness and the choice set are reproduced:

    * ``missed="observed"`` (default): every simulated session has timeouts on exactly the trials the participant missed
      (a *conditional* check - the number of observed outcomes is identical, so reward statistics are computed over the same trials);
    * ``missed="none"``: no timeouts (only appropriate if the observed session has none);
    * ``missed=<float p>``: independent timeouts with probability ``p`` (an explicit timeout *model*).

    ``available`` is the participant's ``(T, K)`` availability schedule, if any; simulated choices respect it.
    """
    rng = as_rng(rng)
    c = np.asarray(choices, dtype=int)
    r = np.asarray(rewards, dtype=float)
    if isinstance(missed, str):
        if missed not in ("observed", "none"):
            raise ValueError("missed must be 'observed', 'none' or a probability")
        mask, prob = ((c == MISSED) if missed == "observed" else None), 0.0
    else:
        mask, prob = None, float(missed)
    stats = statistics or make_statistics(win_threshold)
    obs_tab = _table(c, r, win_threshold)
    observed = {k: f(obs_tab, model.n_actions) for k, f in stats.items()}
    reps: dict[str, list[float]] = {k: [] for k in stats}
    for _ in range(n_sims):
        sim = model.simulate(params, env_factory(), len(c), rng, available=available, missed_prob=prob, missed_mask=mask)
        tab = _table(sim["choices"], sim["rewards"], win_threshold)
        for k, f in stats.items():
            reps[k].append(f(tab, model.n_actions))
    return _summarise(observed, {k: np.array(v) for k, v in reps.items()})


def misfit_rate(
    checks: pd.DataFrame, alpha: float = 0.05, flag_threshold: float = 0.30, level: float = 0.95
) -> pd.DataFrame:
    """Population summary of per-participant checks (long table with ``statistic`` and ``p_value`` columns, optionally ``model``).

    For every (model, statistic): number of participants, fraction with ``p_value < alpha`` (``misfit_rate``), the chance level,
    a Wilson interval for the rate, and ``population_flag`` - ``True`` only if the rate exceeds the *declared* ``flag_threshold``
    (``docs/analysis_plan.md`` §7: 30 %). ``interpretation`` says in words: ``flagged`` (above the threshold), ``elevated, below
    flag threshold`` (credibly above chance but not above the threshold) or ``consistent with chance``.
    """
    from humanchoice.evaluation.metrics import wilson_interval

    d = checks.dropna(subset=["p_value"])
    keys = (["model"] if "model" in d.columns else []) + ["statistic"]
    rows = []
    for key, g in d.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        n, k = len(g), int((g["p_value"] < alpha).sum())
        lo, hi = wilson_interval(k, n, level)
        rate = k / n
        flag = rate > flag_threshold
        interp = "flagged" if flag else ("elevated, below flag threshold" if lo > alpha else "consistent with chance")
        rows.append({**dict(zip(keys, key)), "n_participants": n, "n_misfit": k, "n": n, "misfit_rate": rate,
                     "expected_by_chance": alpha, "ci_lower": lo, "ci_upper": hi, "flag_threshold": flag_threshold,
                     "population_flag": bool(flag), "interpretation": interp})
    return pd.DataFrame(rows)
