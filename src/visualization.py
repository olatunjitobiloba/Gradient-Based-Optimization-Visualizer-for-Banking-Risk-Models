import numpy as np


def compute_vector_field(loss_fn, x_range, y_range, resolution=20, h=1e-5):
    """Compute gradient vectors (U, V) over a meshgrid of (x, y) points.

    Returns X, Y, U, V where each is shape (resolution, resolution).
    U and V are the partial derivatives (d/dx, d/dy) of loss_fn at each grid point.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(resolution):
        for j in range(resolution):
            p = np.array([X[i, j], Y[i, j]], dtype=float)

            p_plus = p.copy()
            p_plus[0] += h
            p_minus = p.copy()
            p_minus[0] -= h
            U[i, j] = (loss_fn(p_plus) - loss_fn(p_minus)) / (2.0 * h)

            p_plus = p.copy()
            p_plus[1] += h
            p_minus = p.copy()
            p_minus[1] -= h
            V[i, j] = (loss_fn(p_plus) - loss_fn(p_minus)) / (2.0 * h)

    return X, Y, U, V
