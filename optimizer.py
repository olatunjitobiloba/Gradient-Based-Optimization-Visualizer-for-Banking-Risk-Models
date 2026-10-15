import numpy as np


def gradient_descent(loss_fn, grad_fn, init_params, lr=0.01, steps=100):
    params = np.array(init_params, dtype=float)
    history = [params.copy()]
    
    for _ in range(steps):
        grad = grad_fn(params)
        params = params - lr * grad
        history.append(params.copy())
        
    return params, history
