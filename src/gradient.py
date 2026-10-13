import numpy as np


class NumericalGradient:
    def __init__(self, func, h=1e-5):
        self.func = func
        self.h = h

    def compute(self, x):
        x = np.array(x, dtype=float)
        grad = np.zeros(len(x))
        for i in range(len(x)):
            x[i] += self.h
            up = self.func(x)
            x[i] -= 2 * self.h
            down = self.func(x)
            x[i] += self.h
            grad[i] = (up - down) / (2 * self.h)
        return grad


def verify_gradient(func, analytical_grad, x, h=1e-5, tol=1e-6):
    numerical = NumericalGradient(func, h).compute(x)
    analytical = analytical_grad(x)
    return np.max(np.abs(numerical - analytical)) < tol
