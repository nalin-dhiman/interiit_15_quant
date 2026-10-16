"""Conditional finite-family response bounds, not estimated-market guarantees.

The caller supplies *uniform, known* derivative bounds on one parameter
interval, with unchanged state/action maps and feasible sets.  The proof is in
docs/model.md.  Float evaluation is not interval arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .mdp import FiniteMDP, fixed_policy_derivative, validate_policy


@dataclass(frozen=True)
class UniformDerivativeBounds:
    """Suprema over all times, states, feasible actions and the entire interval.

    Kernel bounds are row L1 norms. Reward and terminal bounds are sup norms.
    These are mathematical premises; this object cannot establish them from
    observations, fitted standard deviations or a parameter grid.
    """
    kernel_first: float
    kernel_second: float
    reward_abs: float
    reward_first: float
    reward_second: float
    terminal_abs: float
    terminal_first: float = 0.0
    terminal_second: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in vars(self).values()):
            raise ValueError("Every supplied uniform bound must be finite and nonnegative")


def derivative_envelopes(horizon: int, bounds: UniformDerivativeBounds) -> np.ndarray:
    """Rows [remaining steps, (value, first derivative, second derivative)].

    The same envelope bounds a fixed-policy value and an arbitrary feasible
    first action followed by that policy.  The policy must be parameter-fixed.
    """
    if (not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool)
            or horizon < 0):
        raise ValueError("Horizon must be a nonnegative integer")
    out = np.empty((int(horizon) + 1, 3))
    out[0] = (bounds.terminal_abs, bounds.terminal_first, bounds.terminal_second)
    for h in range(1, int(horizon) + 1):
        value, first, second = map(float, out[h - 1])
        out[h] = (bounds.reward_abs + value,
                  bounds.reward_first + bounds.kernel_first * value + first,
                  bounds.reward_second + bounds.kernel_second * value
                  + 2 * bounds.kernel_first * first + second)
    if not np.isfinite(out).all():
        raise ArithmeticError("Derivative envelope exceeds floating-point range")
    return out


def softmax_affine_payoff_bounds(*, direction_span: float,
                                primitive_reward_abs: float,
                                primitive_reward_slope_abs: float,
                                terminal_abs: float,
                                terminal_slope_abs: float = 0.0,
                                terminal_curvature_abs: float = 0.0,
                                ) -> UniformDerivativeBounds:
    """Analytic bounds for a declared finite softmax-tilt primitive family.

    p_theta(j) is proportional to p0(j)*exp(theta*d_j), with fixed payload map
    and strictly positive p0.  The direction range is at most direction_span
    in every row. Primitive rewards are affine in theta, bounded throughout
    the interval by primitive_reward_abs, with slope bound as supplied.
    Terminal bounds must independently hold on that interval.

    Passing small numbers does not verify the premises. In particular the
    primitive reward absolute bound must cover both endpoints, not just theta0.
    """
    values = (direction_span, primitive_reward_abs, primitive_reward_slope_abs,
              terminal_abs, terminal_slope_abs, terminal_curvature_abs)
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Finite nonnegative family bounds required")
    first = direction_span / 2
    second = direction_span * direction_span / 2
    return UniformDerivativeBounds(
        kernel_first=first,
        kernel_second=second,
        reward_abs=primitive_reward_abs,
        reward_first=first * primitive_reward_abs + primitive_reward_slope_abs,
        reward_second=second * primitive_reward_abs
        + 2 * first * primitive_reward_slope_abs,
        terminal_abs=terminal_abs,
        terminal_first=terminal_slope_abs,
        terminal_second=terminal_curvature_abs,
    )


@dataclass(frozen=True)
class GuardedResponseResult:
    policy: np.ndarray
    linear_action_values: np.ndarray
    action_error_bounds: np.ndarray
    value: np.ndarray
    derivative: np.ndarray


def guarded_linear_policy(model: FiniteMDP, baseline: np.ndarray,
                          kernel_derivative: np.ndarray,
                          reward_derivative: np.ndarray,
                          delta: float, bounds: UniformDerivativeBounds, *,
                          terminal_derivative: np.ndarray | None = None,
                          additional_uniform_error: float = 0.0,
                          ) -> GuardedResponseResult:
    """Select only when the linear baseline-Q gap strictly exceeds 2*error.

    The returned policy conditionally improves the baseline in the *known
    perturbed finite family* if all uniform premises hold. ``delta`` must lie in
    the declared family interval; this routine does not know that interval.
    Additional error must itself be a simultaneous bound, not a fitted margin.
    The mathematical statement is in real arithmetic. No floating-point
    certificate or certification of any SDK trading model is supplied here.
    """
    if (not math.isfinite(delta) or not math.isfinite(additional_uniform_error)
            or additional_uniform_error < 0):
        raise ValueError("Finite delta and nonnegative simultaneous error required")
    pi = validate_policy(model, baseline)
    value, derivative = fixed_policy_derivative(
        model, pi, kernel_derivative, reward_derivative, terminal_derivative)
    horizon, states, _, _ = model.P.shape
    envelope = derivative_envelopes(horizon, bounds)
    q0 = model.reward + np.einsum("tsaj,tj->tsa", model.P, value[1:])
    dq0 = (np.asarray(reward_derivative, float)
           + np.einsum("tsaj,tj->tsa", kernel_derivative, value[1:])
           + np.einsum("tsaj,tj->tsa", model.P, derivative[1:]))
    linear = q0 + delta * dq0
    errors = np.array([0.5 * delta * delta * envelope[horizon-t, 2]
                       + additional_uniform_error for t in range(horizon)])
    if not np.isfinite(linear).all() or not np.isfinite(errors).all():
        raise ArithmeticError("Response calculation exceeds floating-point range")
    masked = np.where(model.feasible, linear, -np.inf)
    best = np.argmax(masked, axis=-1)
    proposed = linear[np.arange(horizon)[:, None], np.arange(states), best]
    current = linear[np.arange(horizon)[:, None], np.arange(states), pi]
    choose = proposed - current > 2 * errors[:, None]
    return GuardedResponseResult(np.where(choose, best, pi), linear, errors,
                                 value, derivative)
