"""Tests for the finite difference gradient calculator and its verification."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for location in (ROOT / "src", ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from gradient import NumericalGradient, relative_error, verify_gradient


def quadratic(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def quadratic_grad(x: np.ndarray) -> np.ndarray:
    return 2.0 * x


def tanh_layer(x: np.ndarray) -> float:
    return float(np.sum(np.tanh(x) ** 2))


def tanh_layer_grad(x: np.ndarray) -> np.ndarray:
    activations = np.tanh(x)
    return 2.0 * activations * (1.0 - activations**2)


def test_central_difference_matches_quadratic_gradient():
    x = np.array([0.3, -1.2, 4.5, 9.1])
    result = verify_gradient(quadratic, quadratic_grad, x, tol=1e-6)
    assert result.passed
    assert result.max_relative_error < 1e-6
    assert np.allclose(result.numerical, quadratic_grad(x))


def test_central_difference_matches_composite_layer_gradient():
    x = np.array([-2.0, -0.5, 0.0, 0.7, 3.3])
    result = verify_gradient(tanh_layer, tanh_layer_grad, x, tol=1e-6)
    assert result.passed
    assert result.max_relative_error < 1e-6


def test_central_difference_beats_forward_difference():
    x = np.array([1.5, -0.8, 2.2])
    analytical = quadratic_grad(x)
    central = NumericalGradient(quadratic).compute(x)
    forward = NumericalGradient(quadratic).compute_forward(x)
    assert relative_error(central, analytical) < relative_error(forward, analytical)


def test_gradient_shape_is_preserved():
    x = np.linspace(-3.0, 3.0, 10)
    gradient = NumericalGradient(quadratic).compute(x)
    assert gradient.shape == x.shape


def test_two_dimensional_input_is_supported():
    x = np.array([[0.5, 1.5], [2.5, -3.5]])
    gradient = NumericalGradient(quadratic).compute(x)
    assert gradient.shape == x.shape
    assert np.allclose(gradient, 2.0 * x, atol=1e-4)


def test_verification_detects_an_incorrect_analytical_gradient():
    def wrong_grad(x: np.ndarray) -> np.ndarray:
        return 3.0 * x

    x = np.array([1.0, 2.0, 3.0])
    result = verify_gradient(quadratic, wrong_grad, x, tol=1e-6)
    assert not result.passed
    assert result.max_relative_error > 1e-6


def test_relative_error_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        relative_error(np.zeros(3), np.zeros(4))


def test_negative_step_size_is_rejected():
    with pytest.raises(ValueError):
        NumericalGradient(quadratic, h=-1e-5)


def test_matches_the_project_loss_landscape_analytical_gradient():
    loss_landscape = pytest.importorskip("loss_landscape")
    x = np.array([-4.2, -1.1, 0.0, 0.9, 2.7])
    result = verify_gradient(
        loss_landscape.compute_loss,
        loss_landscape.compute_gradient,
        x,
        tol=1e-5,
    )
    assert result.passed
    assert result.max_relative_error < 1e-5


def test_no_numerical_instability_across_a_range_of_magnitudes():
    for scale in (1e-3, 1.0, 1e3):
        x = scale * np.array([0.4, -1.7, 3.1])
        result = verify_gradient(quadratic, quadratic_grad, x, tol=1e-5)
        assert result.passed
        assert np.all(np.isfinite(result.numerical))
