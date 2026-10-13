import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gradient import NumericalGradient, verify_gradient


def loss(x):
    return np.sum(x**2)


def loss_grad(x):
    return 2 * x


def test_numerical_gradient_matches_analytical():
    x = np.array([1.0, 2.0, 3.0])
    result = NumericalGradient(loss).compute(x)
    assert np.allclose(result, loss_grad(x))


def test_verify_gradient_passes():
    x = np.array([1.0, 2.0, 3.0])
    assert verify_gradient(loss, loss_grad, x)


def test_verify_gradient_detects_wrong_gradient():
    def wrong(x):
        return 3 * x

    x = np.array([1.0, 2.0, 3.0])
    assert not verify_gradient(loss, wrong, x)
