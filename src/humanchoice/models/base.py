"""Common interface for trial-by-trial computational models of choice among ``K`` options.

A :class:`ChoiceModel` is a *stateless* description: all learned quantities live in an explicit
``state`` object that is threaded through ``update``. This keeps likelihood evaluation, simulation
and visualisation on exactly the same code path (a frequent source of bugs in modelling code).

Conventions
-----------
* Actions are integer indices ``0 .. K-1``; ``-1`` (:data:`humanchoice.utils.MISSED`) marks a trial on which no
  choice was made. Missed trials carry no outcome: the model state is *not* updated and the trial does not enter
  the likelihood.
* An optional boolean ``available`` array of shape ``(T, K)`` restricts the choice set on each trial; probabilities
  are exactly zero for unavailable actions and a chosen action must be available.
* Unless a model overrides :meth:`choice_probs`, choices follow a softmax::

      P(a_t) = exp(beta * V_t(a)) / sum_b exp(beta * V_t(b))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from humanchoice.utils import EPS, MISSED, as_rng, masked_softmax


class ChoiceModel(ABC):
    """Abstract base class for learning-and-choice models over ``n_actions`` options."""

    name: str = "model"
    param_names: tuple[str, ...] = ()
    #: hard optimisation bounds for each parameter
    bounds: tuple[tuple[float, float], ...] = ()
    #: ranges used to draw synthetic participants and optimiser starting points
    sample_ranges: tuple[tuple[float, float], ...] = ()
    #: ``"any"`` (finite real rewards) or ``"binary"`` (rewards must be 0/1, e.g. Beta-Bernoulli models)
    reward_support: str = "any"
    #: set for models that only make sense for a specific number of options (e.g. two-option reversal learning)
    fixed_n_actions: int | None = None
    n_actions: int = 2

    def __init__(self, n_actions: int = 2) -> None:
        if isinstance(n_actions, bool) or not isinstance(n_actions, (int, np.integer)) or n_actions < 2:
            raise ValueError("n_actions must be an integer >= 2")
        if self.fixed_n_actions is not None and n_actions != self.fixed_n_actions:
            raise ValueError(f"{self.name} is defined for exactly {self.fixed_n_actions} options, got n_actions={n_actions}")
        self.n_actions = int(n_actions)

    # ------------------------------------------------------------- parameters
    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def to_array(self, params: dict[str, float] | ArrayLike) -> NDArray[np.float64]:
        """Convert a ``{name: value}`` dict (or sequence) into a validated parameter vector."""
        if isinstance(params, dict):
            missing = set(self.param_names) - set(params)
            if missing:
                raise ValueError(f"missing parameters: {sorted(missing)}")
            arr = np.array([params[n] for n in self.param_names], dtype=float)
        else:
            arr = np.asarray(params, dtype=float)
        if arr.shape != (self.n_params,):
            raise ValueError(f"{self.name} expects {self.n_params} parameters {self.param_names}, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("parameters must be finite")
        for v, n, (lo, hi) in zip(arr, self.param_names, self.bounds):
            if not lo - 1e-9 <= v <= hi + 1e-9:
                raise ValueError(f"parameter {n}={v} outside bounds [{lo}, {hi}]")
        return arr

    def to_dict(self, params: ArrayLike) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.param_names, np.asarray(params))}

    def sample_params(self, rng: int | np.random.Generator | None = None) -> NDArray[np.float64]:
        """Draw a parameter vector uniformly from ``sample_ranges``."""
        rng = as_rng(rng)
        return np.array([rng.uniform(lo, hi) for lo, hi in self.sample_ranges])

    @property
    def beta_index(self) -> int:
        return self.param_names.index("beta")

    # ------------------------------------------------------------ model core
    @abstractmethod
    def init_state(self, params: NDArray) -> Any:
        """Initial latent state before the first trial."""

    @abstractmethod
    def action_values(self, params: NDArray, state: Any) -> NDArray[np.float64]:
        """Latent value of each action given the current state (used for plotting and the default softmax)."""

    @abstractmethod
    def update(self, params: NDArray, state: Any, action: int, reward: float) -> Any:
        """Return the new state after observing ``reward`` for ``action``."""

    def choice_probs(self, params: NDArray, state: Any, available: ArrayLike | None = None) -> NDArray[np.float64]:
        """Probability of each action; a softmax over :meth:`action_values` unless overridden."""
        return masked_softmax(self.action_values(params, state), beta=params[self.beta_index], available=available)

    def nested_start(
        self, choices: ArrayLike, rewards: ArrayLike, available: ArrayLike | None, rng: np.random.Generator
    ) -> NDArray[np.float64] | None:
        """Optional extra optimiser starting point derived from a simpler model this model *nests*.

        Returning the nested model's maximum-likelihood solution (mapped into this model's parameter space) guarantees that
        the richer model cannot end up with a worse likelihood than the model it contains, which multi-start optimisation
        from random points does not. ``None`` (default) means "no nested model".
        """
        return None

    # ------------------------------------------------------------ validation
    def check_rewards(self, rewards: ArrayLike, choices: ArrayLike | None = None) -> None:
        """Raise if ``rewards`` are outside this model's ``reward_support`` (missed trials are ignored)."""
        if self.reward_support == "any":
            return
        r = np.asarray(rewards, dtype=float)
        if choices is not None:
            r = r[np.asarray(choices) != MISSED]
        if not np.isin(r, (0.0, 1.0)).all():
            raise ValueError(f"{self.name} requires binary (0/1) rewards")

    def _prepare(
        self, choices: ArrayLike, rewards: ArrayLike, available: ArrayLike | None
    ) -> tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.bool_] | None]:
        c = np.asarray(choices)
        if c.dtype.kind == "f":
            if not np.array_equal(c, np.round(c)):
                raise ValueError("choices must be integer action indices")
        elif c.dtype.kind not in "iub":
            raise ValueError("choices must be integer action indices")
        c = c.astype(int)
        r = np.asarray(rewards, dtype=float)
        if c.shape != r.shape or c.ndim != 1:
            raise ValueError("choices and rewards must be 1-D arrays of equal length")
        if c.size and (c.min() < MISSED or c.max() >= self.n_actions):
            raise ValueError(f"choices must be action indices in [0, {self.n_actions}) or {MISSED} (missed trial)")
        self.check_rewards(r, c)
        avail = None
        if available is not None:
            avail = np.asarray(available, dtype=bool)
            if avail.shape != (len(c), self.n_actions):
                raise ValueError(f"available must have shape ({len(c)}, {self.n_actions}), got {avail.shape}")
            if not avail.any(axis=1).all():
                raise ValueError("at least one action must be available on every trial")
            obs = c >= 0
            if not avail[np.flatnonzero(obs), c[obs]].all():
                raise ValueError("a chosen action was marked as unavailable on that trial")
        return c, r, avail

    # ----------------------------------------------------------- derived API
    def trajectory(
        self,
        params: ArrayLike,
        choices: ArrayLike,
        rewards: ArrayLike,
        available: ArrayLike | None = None,
    ) -> dict[str, NDArray]:
        """Replay a choice/reward sequence.

        Returns ``probs`` (T, K) and ``values`` (T, K) *before* each trial (one-step-ahead predictions) and
        ``observed`` (T,) - ``False`` on missed trials, on which the state is left unchanged.
        """
        p = self.to_array(params)
        c, r, avail = self._prepare(choices, rewards, available)
        T = len(c)
        probs = np.empty((T, self.n_actions))
        values = np.empty((T, self.n_actions))
        state = self.init_state(p)
        for t in range(T):
            values[t] = self.action_values(p, state)
            probs[t] = self.choice_probs(p, state, None if avail is None else avail[t])
            if c[t] != MISSED:
                state = self.update(p, state, int(c[t]), float(r[t]))
        return {"probs": probs, "values": values, "observed": c != MISSED}

    def log_likelihood(
        self, params: ArrayLike, choices: ArrayLike, rewards: ArrayLike, available: ArrayLike | None = None
    ) -> float:
        """``sum_t log P(choice_t | history_t, theta)`` over observed (non-missed) trials."""
        c = np.asarray(choices, dtype=int)
        traj = self.trajectory(params, choices, rewards, available)
        obs = traj["observed"]
        chosen = traj["probs"][np.flatnonzero(obs), c[obs]]
        return float(np.log(np.clip(chosen, EPS, None)).sum())

    def simulate(
        self,
        params: ArrayLike,
        env,
        n_trials: int,
        rng: int | np.random.Generator | None = None,
        available: ArrayLike | None = None,
        missed_prob: float = 0.0,
        missed_mask: ArrayLike | None = None,
    ) -> dict[str, NDArray]:
        """Generate behaviour by letting the model interact with ``env`` (which must offer the same number of options).

        With ``missed_prob > 0`` each trial is, independently, a *timeout*: the choice is recorded as ``-1`` with reward 0,
        the model state is not updated, and the environment still advances (its payoffs drift on regardless), mirroring
        tasks in which a missed response simply forfeits the trial. Alternatively ``missed_mask`` (boolean, length ``n_trials``) fixes
        *which* trials are missed - e.g. a participant's actual timeout pattern for a conditional check; it excludes ``missed_prob``.
        """
        if not 0.0 <= missed_prob < 1.0:
            raise ValueError("missed_prob must lie in [0, 1)")
        mask = None
        if missed_mask is not None:
            if missed_prob > 0.0:
                raise ValueError("give either missed_prob or missed_mask, not both")
            mask = np.asarray(missed_mask, dtype=bool)
            if mask.shape != (n_trials,):
                raise ValueError(f"missed_mask must have shape ({n_trials},)")
        env_k = getattr(env, "n_actions", self.n_actions)
        if env_k != self.n_actions:
            raise ValueError(f"environment offers {env_k} options but the model has n_actions={self.n_actions}")
        rng = as_rng(rng)
        p = self.to_array(params)
        avail = None
        if available is not None:
            avail = np.asarray(available, dtype=bool)
            if avail.shape != (n_trials, self.n_actions) or not avail.any(axis=1).all():
                raise ValueError("available must have shape (n_trials, n_actions) with >= 1 available action per trial")
        env.reset(rng)
        state = self.init_state(p)
        choices = np.empty(n_trials, dtype=int)
        rewards = np.empty(n_trials, dtype=float)
        probs = np.empty((n_trials, self.n_actions))
        values = np.empty((n_trials, self.n_actions))
        for t in range(n_trials):
            values[t] = self.action_values(p, state)
            probs[t] = self.choice_probs(p, state, None if avail is None else avail[t])
            if (mask[t] if mask is not None else (missed_prob > 0.0 and rng.random() < missed_prob)):
                choices[t], rewards[t] = MISSED, 0.0
                env.step(0, rng)  # the world moves on; the (unobserved) outcome is discarded
                continue
            choices[t] = rng.choice(self.n_actions, p=probs[t])
            rewards[t] = env.step(int(choices[t]), rng)
            state = self.update(p, state, int(choices[t]), rewards[t])
        return {"choices": choices, "rewards": rewards, "probs": probs, "values": values}
