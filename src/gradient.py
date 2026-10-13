"""Numerical differentiation utilities for the loss landscape.

This module provides a finite difference gradient calculator and a helper
that verifies it against an analytical (hand derived) gradient. The central
difference formula ``(f(x + h) - f(x - h)) / 2h`` is the default because it
has a lower truncation error than the forward difference
``(f(x + h) - f(x)) / h`` for the same step size ``h``.
"""

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

ArrayLike = Sequence[float] | np.ndarray
ScalarFunction = Callable[[np.ndarray], float]
GradientFunction = Callable[[np.ndarray], ArrayLike]

_EPSILON = 1e-12


class NumericalGradient:
    """Approximate the gradient of a scalar function with finite differences."""

    def __init__(self, func: ScalarFunction, h: float = 1e-5) -> None:
        if h <= 0:
            raise ValueError("step size h must be positive")
        self.func = func
        self.h = h

    def compute(self, x: ArrayLike) -> np.ndarray:
        """Return the central difference gradient at ``x``.

        Uses ``(f(x + h) - f(x - h)) / 2h`` which is accurate to ``O(h^2)``
        and avoids the first order bias of the forward difference.
        """
        x = np.asarray(x, dtype=float)
        gradient = np.zeros_like(x)
        for index in np.ndindex(x.shape):
            original = x[index]
            x[index] = original + self.h
            forward = self.func(x)
            x[index] = original - self.h
            backward = self.func(x)
            x[index] = original
            gradient[index] = (forward - backward) / (2.0 * self.h)
        return gradient

    def compute_forward(self, x: ArrayLike) -> np.ndarray:
        """Return the forward difference gradient at ``x``.

        Uses ``(f(x + h) - f(x)) / h`` which is accurate to ``O(h)``.
        Provided mainly for comparing approximation error against the
        central difference implementation.
        """
        x = np.asarray(x, dtype=float)
        base = self.func(x)
        gradient = np.zeros_like(x)
        for index in np.ndindex(x.shape):
            original = x[index]
            x[index] = original + self.h
            gradient[index] = (self.func(x) - base) / self.h
            x[index] = original
        return gradient


class GradientVerification(NamedTuple):
    """Result of comparing a numerical gradient to an analytical gradient."""

    passed: bool
    max_relative_error: float
    tolerance: float
    numerical: np.ndarray
    analytical: np.ndarray


def relative_error(numerical: ArrayLike, analytical: ArrayLike) -> float:
    """Return the largest relative error between two gradient vectors.

    The relative error is ``|num - ana| / max(|num| + |ana|, eps)`` which
    stays well defined even when either gradient component is zero.
    """
    numerical = np.asarray(numerical, dtype=float)
    analytical = np.asarray(analytical, dtype=float)
    if numerical.shape != analytical.shape:
        raise ValueError("numerical and analytical gradients must share a shape")
    denominator = np.maximum(np.abs(numerical) + np.abs(analytical), _EPSILON)
    return float(np.max(np.abs(numerical - analytical) / denominator))


def verify_gradient(
    func: ScalarFunction,
    analytical_grad: GradientFunction,
    x: ArrayLike,
    h: float = 1e-5,
    tol: float = 1e-6,
) -> GradientVerification:
    """Compare a numerical gradient against a manually derived gradient.

    Returns a :class:`GradientVerification` whose ``passed`` flag is ``True``
    when every component is within ``tol`` of the analytical gradient.
    """
    x = np.asarray(x, dtype=float)
    numerical = NumericalGradient(func, h).compute(x)
    analytical = np.asarray(analytical_grad(x), dtype=float)
    error = relative_error(numerical, analytical)
    return GradientVerification(
        passed=error <= tol,
        max_relative_error=error,
        tolerance=tol,
        numerical=numerical,
        analytical=analytical,
    )
