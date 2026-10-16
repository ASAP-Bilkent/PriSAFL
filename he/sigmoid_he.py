import numpy as np
from he.prisafltext import PriSAFLText
from he.poly_eval import eval_poly_he

SIGMOID_TAU = 4.0

def _get_sigmoid_coeffs(poly_degree, interval, n_grid=4001):
    from scipy.optimize import linprog
    z = np.linspace(-1, 1, n_grid)
    s = 1.0 / (1.0 + np.exp(-z * interval / SIGMOID_TAU))
    V = np.vander(z, poly_degree + 1, increasing=True)
    D = np.hstack([np.zeros((n_grid, 1)), V[:, :-1] * np.arange(1, poly_degree + 1)])
    one, zero = np.ones((n_grid, 1)), np.zeros((n_grid, 1))
    A = np.vstack([np.hstack([V, -one]), np.hstack([-V, -one]), np.hstack([-V, zero]),
                   np.hstack([V, zero]), np.hstack([-D, zero])])
    b = np.concatenate([s, -s, np.zeros(n_grid), np.ones(n_grid), np.zeros(n_grid)])
    r = linprog(np.r_[np.zeros(poly_degree + 1), 1.0], A_ub=A, b_ub=b,
                bounds=[(None, None)] * (poly_degree + 2), method="highs")
    if not r.success:
        raise RuntimeError(f"sigmoid fit failed: {r.message}")
    coeffs = r.x[:-1]
    return {i: float(c) for i, c in enumerate(coeffs) if abs(c) > 1e-12}

def _scaled_sigmoid_coeffs(poly_degree, interval):
    coeffs = _get_sigmoid_coeffs(poly_degree, interval)
    scaled_coeffs = {}
    for power, coeff in coeffs.items():
        if power == 0:
            scaled_coeffs[power] = coeff
        else:
            scaled_coeffs[power] = coeff / (interval ** power)
    return scaled_coeffs

def sigmoid_he(x, interval=15.0, poly_degree=3):
    scaled_coeffs = _scaled_sigmoid_coeffs(poly_degree, interval)

    result = eval_poly_he(x, scaled_coeffs)
    if PriSAFLText.VERBOSE >= 1 and result._cipher_data is not None:
        print(f"    [level] sigmoid_he output: {result._cipher_data.GetLevel()}")
    return result
