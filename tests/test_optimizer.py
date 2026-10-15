import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer import gradient_descent


def loss(x):
    return np.sum(x**2)


def grad(x):
    return 2 * x


def test_converges_to_minimum_of_convex_function():
    params, _history = gradient_descent(loss, grad, [5.0, -3.0], lr=0.1, steps=100)
    assert np.allclose(params, [0.0, 0.0], atol=1e-4)


def test_history_records_initial_and_every_step():
    _params, history = gradient_descent(loss, grad, [1.0], lr=0.1, steps=10)
    assert len(history) == 11


def test_loss_decreases_each_step():
    _params, history = gradient_descent(loss, grad, [5.0], lr=0.1, steps=20)
    losses = [loss(p) for p in history]
    assert losses[-1] < losses[0]
