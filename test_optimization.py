import numpy as np
from loss_landscape import compute_gradient, compute_loss


def test_loss_dimensions():
    assert isinstance(compute_loss(np.zeros(5)), float)


def test_gradient_dimensions():
    assert compute_gradient(np.zeros(5)).shape == (5,)


def test_loss_known_values():
    assert np.allclose(compute_loss(np.zeros(2)), 0.0)
    assert np.allclose(compute_loss(np.array([1.0])), 1.0)
    assert np.allclose(compute_loss(np.array([2.0])), 4.0)


def test_gradient_known_values():
    assert np.allclose(compute_gradient(np.zeros(2)), [0.0, 0.0])
    assert np.allclose(compute_gradient(np.array([1.0])), [2.0])
