"""Regular finite-primitive response of baseline-continuation reset values.

Known synthetic family only: fixed primitive payloads/actions/ticks/targets,
softmax-tilted positive atom probabilities, and an affine actual exchange fee.
This does not identify a competition kernel or certify an empirical trust region.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math
import numpy as np

from .continuation import (Action, BandPolicy, Batch, FiniteRFQModel,
                           ImprovedPolicy, Policy, initial_weights)


class PrimitiveTiltFamily:
    """Common-support primitive history; fee is a parameter-dependent payoff.

    Batch transition and every payload/quantity/target/tick stay fixed. Positive
    base atom probabilities are required. No derivative of deterministic cash
    path densities is used. Parameter theta is dimensionless.
    """
    kind = "atom_probability_and_exchange_fee"

    def __init__(self, base: FiniteRFQModel, directions, fee_derivative: float):
        self.base = base
        self.directions = tuple(tuple(float(v) for v in row) for row in directions)
        self.fee_derivative = float(fee_derivative)
        if len(self.directions) != len(base.batches) or not math.isfinite(self.fee_derivative):
            raise ValueError("Finite fee derivative and one atom direction row per batch required")
        for batch, row in zip(base.batches, self.directions):
            if len(row) != len(batch.atoms) or not all(math.isfinite(v) for v in row):
                raise ValueError("Finite directions must align with primitive atoms")
            if any(atom.probability <= 0 for atom in batch.atoms):
                raise ValueError("Regular response requires strictly positive primitive support")

    def probabilities(self, b: int, theta: float = 0.) -> tuple[np.ndarray, np.ndarray]:
        if not math.isfinite(theta):
            raise ValueError("Finite perturbation required")
        p0 = np.array([a.probability for a in self.base.batches[b].atoms])
        direction = np.asarray(self.directions[b])
        logits = np.log(p0)+theta*direction
        if not np.isfinite(logits).all():
            raise ValueError("Perturbation exceeds numerical likelihood domain")
        weights = np.exp(logits-logits.max())
        p = weights/weights.sum()
        if np.any(p == 0):
            raise ValueError("Numerical underflow changed regular support")
        dp = p*(direction-float(p@direction))
        return p, dp

    def model(self, theta: float) -> FiniteRFQModel:
        fee = self.base.config.exchange_fee+theta*self.fee_derivative
        if not math.isfinite(fee) or fee < 0:
            raise ValueError("Perturbed actual exchange fee must remain nonnegative")
        batches = []
        for b, batch in enumerate(self.base.batches):
            p, _ = self.probabilities(b, theta)
            batches.append(Batch(batch.rfqs, tuple(replace(a, probability=float(prob))
                                                   for a, prob in zip(batch.atoms, p))))
        return FiniteRFQModel(replace(self.base.config, exchange_fee=fee),
                              tuple(batches), self.base.batch_transition.copy())


class TangentPolicyEvaluation:
    """Exact finite recursion for (V^baseline, dV^baseline) at theta=0.

    The policy is held fixed; action selection/argmax is never differentiated.
    include_payoff_derivative=False is a deliberately incomplete diagnostic.
    """
    def __init__(self, family: PrimitiveTiltFamily, policy: Policy, *,
                 include_payoff_derivative: bool = True):
        self.family, self.model, self.policy = family, family.base, policy
        self.include_payoff_derivative = include_payoff_derivative
        self.action_value_calls = self.atom_evaluations = self.state_evaluations = 0
        self._probabilities = tuple(family.probabilities(b) for b in range(len(self.model.batches)))
        self.value = lru_cache(maxsize=None)(self._value)

    def _value(self, t: int, q: int, b: int) -> tuple[float, float]:
        self.state_evaluations += 1
        if t == self.model.config.horizon:
            return self.model.terminal(q), 0.
        if not 0 <= t < self.model.config.horizon:
            raise ValueError("Value time outside horizon")
        return self.action_value(t, q, b, self.policy(t, q, b))

    def action_value(self, t: int, q: int, b: int, action: Action) -> tuple[float, float]:
        if not 0 <= t < self.model.config.horizon:
            raise ValueError("Action time outside horizon")
        self.action_value_calls += 1
        outcomes = self.model.outcomes(q, b, action)
        p, dp = self._probabilities[b]
        value = derivative = 0.
        for outcome, prob, dprob in zip(outcomes, p, dp):
            self.atom_evaluations += 1
            future = dfuture = 0.
            for nb, weight in enumerate(self.model.batch_transition[b]):
                if weight:
                    v, dv = self.value(t+1, outcome.next_inventory, nb)
                    future += float(weight)*v
                    dfuture += float(weight)*dv
            explicit = (-outcome.exchange_quantity*self.family.fee_derivative
                        if self.include_payoff_derivative else 0.)
            total = outcome.objective+future
            value += float(prob)*total
            derivative += float(dprob)*total+float(prob)*(explicit+dfuture)
        return value, derivative

    def initial_value(self, q0: int = 0, initial_batch=None) -> tuple[float, float]:
        values = [self.value(0, q0, b) for b in range(len(self.model.batches))]
        weights = initial_weights(self.model, initial_batch)
        return tuple(sum(float(p)*value[j] for p, value in zip(weights, values))
                     for j in range(2))


class LinearResponsePolicy:
    """Synthetic R3 candidate selector using Q0^baseline+theta*dQ0^baseline.

    Radius is a predeclared empirical domain, not a uniform Taylor certificate.
    Invalid/unsupported/out-of-domain perturbations return the conventional
    baseline. Candidates and baseline are all feasible under the unchanged
    units, quotes and limits. No unknown-market safety claim is attached.
    """
    def __init__(self, family: PrimitiveTiltFamily, baseline: BandPolicy,
                 theta: float, *, trust_radius: float = .15,
                 perturbation_kind: str = PrimitiveTiltFamily.kind):
        if not math.isfinite(trust_radius) or trust_radius < 0:
            raise ValueError("Finite nonnegative empirical radius required")
        self.family, self.baseline, self.theta = family, baseline, float(theta)
        self.model = family.base
        self.baseline_policy = baseline.bind(self.model)
        self.candidate_policy = ImprovedPolicy(self.model, baseline)
        self.supported = (math.isfinite(self.theta) and abs(self.theta) <= trust_radius
                          and perturbation_kind == family.kind)
        if self.supported:
            try:
                family.model(self.theta)  # validate fee and numerical support
            except ValueError:
                self.supported = False
        self.evaluation = TangentPolicyEvaluation(family, self.baseline_policy)
        self.candidate_evaluations = self.fallback_states = 0
        self.__call_cached = lru_cache(maxsize=None)(self._choose)

    def _choose(self, t: int, q: int, b: int) -> Action:
        if not self.supported:
            self.fallback_states += 1
            return self.baseline_policy(t, q, b)
        actions = self.candidate_policy.candidates(t, q, b)
        best, best_value = actions[0], -math.inf
        for action in actions:
            value, derivative = self.evaluation.action_value(t, q, b, action)
            self.candidate_evaluations += 1
            proposal = value+self.theta*derivative
            if proposal > best_value+1e-12:
                best, best_value = action, proposal
        return best

    def __call__(self, t: int, q: int, b: int) -> Action:
        return self.__call_cached(t, q, b)


def eligibility_tilt_directions(model: FiniteRFQModel) -> tuple[tuple[float, ...], ...]:
    """Fixed synthetic perturbation: favour higher-limit joint primitive atoms.

    This is a chosen family definition, not an inferred customer-limit model.
    Latent atoms are available to offline known-model evaluation only.
    """
    return tuple(tuple((sum(a.limit_edges)/len(a.limit_edges)-.1)/.08
                       if a.limit_edges else 0. for a in batch.atoms)
                 for batch in model.batches)
