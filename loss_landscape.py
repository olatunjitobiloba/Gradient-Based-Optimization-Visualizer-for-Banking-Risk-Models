import numpy as np


def compute_loss(weights: np.ndarray) -> float:
    """Compute the non-convex loss value for a given weight vector."""
    A = 10.0
    n = len(weights)
    return A * n + np.sum(weights**2 - A * np.cos(2 * np.pi * weights))


def compute_gradient(weights: np.ndarray) -> np.ndarray:
    """Compute the analytical gradient of the loss function with respect to weights."""
    A = 10.0
    return 2 * weights + 2 * np.pi * A * np.sin(2 * np.pi * weights)
