import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visualization import compute_vector_field


def quadratic(x):
    return float(np.sum(x**2))


def test_vector_field_shapes():
    X, Y, U, V = compute_vector_field(quadratic, (-2.0, 2.0), (-2.0, 2.0), resolution=10)
    assert X.shape == (10, 10)
    assert Y.shape == (10, 10)
    assert U.shape == (10, 10)
    assert V.shape == (10, 10)


def test_vector_field_matches_analytical_gradient():
    X, Y, U, V = compute_vector_field(quadratic, (-2.0, 2.0), (-2.0, 2.0), resolution=5)
    assert np.allclose(U, 2 * X, atol=1e-3)
    assert np.allclose(V, 2 * Y, atol=1e-3)


def test_vector_field_default_resolution_is_20():
    X, _Y, U, V = compute_vector_field(quadratic, (-1.0, 1.0), (-1.0, 1.0))
    assert X.shape == (20, 20)
    assert U.shape == (20, 20)
    assert V.shape == (20, 20)
