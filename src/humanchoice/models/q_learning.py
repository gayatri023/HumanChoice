"""Model-free temporal-difference learners for ``K`` options (the Q-learning family).

All variants share one implementation, :class:`_QFamily`, parameterised by three switches:

* ``asymmetric`` - separate learning rates for positive / negative prediction errors;
* ``sticky``     - a perseveration bonus ``kappa`` added to the logit of the previously chosen option;
* ``lapse``      - with probability ``epsilon`` the choice is uniformly random.

Decision rule::

    p_softmax(a) ∝ exp(beta * Q(a) + kappa * 1[a == previous choice])
    P(a)         = (1 - epsilon) * p_softmax(a) + epsilon / K_available

Parameter order is always ``alpha`` (or ``alpha_pos, alpha_neg``), ``beta``, then ``kappa`` and/or ``epsilon``.
Rewards are assumed to lie roughly in [0, 1] (rescale other reward scales at the data layer).
"""

from __future__ import annotations

from math import exp, log

import numpy as np
from numpy.typing import ArrayLike, NDArray

from humanchoice.models.base import ChoiceModel
from humanchoice.utils import EPS, MISSED, masked_softmax


class _QFamily(ChoiceModel):
    asymmetric = False
    sticky = False
    lapse = False

    def __init__(self, n_actions: int = 2, q0: float = 0.5) -> None:
        super().__init__(n_actions)
        self.q0 = float(q0)
        names, bounds, ranges = [], [], []
        if self.asymmetric:
            names += ["alpha_pos", "alpha_neg"]
            bounds += [(0.0, 1.0), (0.0, 1.0)]
            ranges += [(0.05, 0.95), (0.05, 0.95)]
        else:
            names += ["alpha"]
            bounds += [(0.0, 1.0)]
            ranges += [(0.05, 0.95)]
        names += ["beta"]
        bounds += [(0.0, 20.0)]
        ranges += [(0.5, 8.0)]
        if self.sticky:
            names += ["kappa"]
            bounds += [(-2.0, 5.0)]
            ranges += [(0.0, 2.0)]
        if self.lapse:
            names += ["epsilon"]
            bounds += [(0.0, 0.5)]
            ranges += [(0.01, 0.3)]
        # instance-level (the parameter set depends on the switches)
        self.param_names, self.bounds, self.sample_ranges = tuple(names), tuple(bounds), tuple(ranges)

    # ---------------------------------------------------------------- model
    def init_state(self, params: NDArray):
        q = np.full(self.n_actions, self.q0)
        return (q, MISSED) if self.sticky else q

    def action_values(self, params: NDArray, state) -> NDArray[np.float64]:
        return state[0] if self.sticky else state

    def choice_probs(self, params: NDArray, state, available: ArrayLike | None = None) -> NDArray[np.float64]:
        q, last = state if self.sticky else (state, MISSED)
        logits = params[self.param_names.index("beta")] * q
        if self.sticky and last != MISSED:
            logits = logits.copy()
            logits[last] += params[self.param_names.index("kappa")]
        p = masked_softmax(logits, 1.0, available)
        if self.lapse:
            eps = params[self.param_names.index("epsilon")]
            avail = np.ones(self.n_actions) if available is None else np.asarray(available, dtype=float)
            p = (1.0 - eps) * p + eps * avail / avail.sum()
        return p

    def update(self, params: NDArray, state, action: int, reward: float):
        q = (state[0] if self.sticky else state).copy()
        delta = reward - q[action]
        rate = params[0] if (not self.asymmetric or delta >= 0) else params[1]
        q[action] += rate * delta
        return (q, action) if self.sticky else q

    def nested_start(self, choices, rewards, available, rng):
        """Start from the plain Q-learning solution (``alpha+ = alpha-``, ``kappa = 0``, ``epsilon = 0``)."""
        if not (self.asymmetric or self.sticky or self.lapse):
            return None
        from humanchoice.fitting.estimation import fit_model  # local import: fitting depends on models

        base = fit_model(QLearning(self.n_actions, self.q0), choices, rewards, n_starts=10, seed=rng, available=available)
        a, b = base.params["alpha"], base.params["beta"]
        x = [a, a] if self.asymmetric else [a]
        x.append(b)
        if self.sticky:
            x.append(0.0)
        if self.lapse:
            x.append(0.0)
        return np.array(x)

    # ------------------------------------------------- fast likelihood path
    def log_likelihood(
        self, params: ArrayLike, choices: ArrayLike, rewards: ArrayLike, available: ArrayLike | None = None
    ) -> float:
        """Same value as the generic replay, but ~5x faster (scalar loop, incremental softmax denominators)."""
        if available is not None:
            return super().log_likelihood(params, choices, rewards, available)
        p = self.to_array(params)
        c, r, _ = self._prepare(choices, rewards, None)
        if self.asymmetric:
            a_pos, a_neg, beta = float(p[0]), float(p[1]), float(p[2])
            i = 3
        else:
            a_pos = a_neg = float(p[0])
            beta = float(p[1])
            i = 2
        kappa = float(p[i]) if self.sticky else 0.0
        i += 1 if self.sticky else 0
        eps = float(p[i]) if self.lapse else 0.0
        K = self.n_actions
        eps_k = eps / K
        e_kappa = exp(kappa)
        try:
            q = [self.q0] * K
            e = [exp(beta * x) for x in q]
            last, ll = MISSED, 0.0
            for ch, rw in zip(c.tolist(), r.tolist()):
                if ch == MISSED:
                    continue
                total = sum(e)
                if self.sticky and last != MISSED:
                    denom = total + e[last] * (e_kappa - 1.0)
                    num = e[ch] * (e_kappa if ch == last else 1.0)
                else:
                    denom, num = total, e[ch]
                pr = (1.0 - eps) * num / denom + eps_k
                ll += log(max(EPS, pr))
                delta = rw - q[ch]
                q[ch] += (a_pos if delta >= 0.0 else a_neg) * delta
                e[ch] = exp(beta * q[ch])
                last = ch
            return ll
        except (OverflowError, ZeroDivisionError):  # extreme value scales: use the numerically stable generic path
            return super().log_likelihood(params, choices, rewards, available)


class QLearning(_QFamily):
    """Rescorla-Wagner / Q-learning: ``Q(a) <- Q(a) + alpha * (r - Q(a))``; unchosen options are not updated.

    Parameters: ``alpha`` (learning rate, [0, 1]) and ``beta`` (inverse temperature).
    """

    name = "q_learning"


class AsymmetricQLearning(_QFamily):
    """Q-learning with separate learning rates for positive and negative prediction errors.

    ``delta = r - Q(a)``; ``Q(a) += alpha_pos * delta`` if ``delta >= 0`` else ``alpha_neg * delta``. Addresses RQ1
    (how do individuals learn from positive vs negative feedback?) and nests :class:`QLearning`.
    """

    name = "asymmetric_q_learning"
    asymmetric = True


class StickyQLearning(_QFamily):
    """Q-learning plus choice perseveration ``kappa`` (> 0: repeat the previous choice; < 0: alternate).

    Perseveration and slow learning can mimic each other, so ``kappa`` must be checked for identifiability
    (see the recovery experiments) before it is interpreted.
    """

    name = "sticky_q_learning"
    sticky = True


class LapseQLearning(_QFamily):
    """Q-learning plus a lapse rate ``epsilon``: with that probability the choice ignores value entirely."""

    name = "lapse_q_learning"
    lapse = True


class FullQLearning(_QFamily):
    """Asymmetric + sticky + lapse Q-learning (the most flexible member of the family)."""

    name = "full_q_learning"
    asymmetric = sticky = lapse = True
