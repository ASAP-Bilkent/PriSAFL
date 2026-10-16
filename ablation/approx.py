import torch
import numpy as np

from activation import _remez_fit

def constrained_fit(deg, B, n=4001, mode='c'):
    from scipy.optimize import linprog
    z = np.linspace(-1, 1, n); s = 1 / (1 + np.exp(-z * B))
    V = np.vander(z, deg + 1, increasing=True)
    D = np.hstack([np.zeros((n, 1)), V[:, :-1] * np.arange(1, deg + 1)])
    one = np.ones((n, 1)); zero = np.zeros((n, 1))
    rows = [(np.hstack([V, -one]), s), (np.hstack([-V, -one]), -s), (np.hstack([-V, zero]), np.zeros(n))]
    if mode == 'c': rows.append((np.hstack([V, zero]), np.ones(n)))
    if mode in ('c', 'pm'): rows.append((np.hstack([-D, zero]), np.zeros(n)))
    A = np.vstack([r[0] for r in rows]); b = np.concatenate([r[1] for r in rows])
    r = linprog(np.r_[np.zeros(deg + 1), 1], A_ub=A, b_ub=b, bounds=[(None, None)] * (deg + 2), method='highs')
    return r.x[:-1]

def _make_approx_sigmoid(spec):
    if spec.startswith("P"):
        coeffs, interval = [float(c) for c in spec[1:].split("_")], 15.0
        spec = ""
    spec, _, temp = spec.partition("T")
    temp = float(temp) if temp else 1.0
    if spec == "exact":
        return lambda x: torch.sigmoid(x / temp)
    mode = spec.lstrip("0123456789-.")
    constrained = bool(mode)
    parts = spec[:len(spec) - len(mode)].split("-") if spec else None
    degree = int(parts[0]) if parts else None
    interval = float(parts[1]) if parts else interval

    if not parts:
        pass
    elif constrained:
        coeffs = constrained_fit(degree, interval / temp, mode=mode)
    else:
        coeffs = _remez_fit(lambda z: 1.0 / (1.0 + np.exp(-(z * interval / temp))),
                            degree, -1.0, 1.0)

    coeffs_t = torch.tensor(coeffs, dtype=torch.float32)
    inv_interval = 1.0 / interval

    def approx_sigmoid(x):
        z = x * inv_interval
        result = coeffs_t[-1].to(x.device)
        for i in range(len(coeffs_t) - 2, -1, -1):
            result = result * z + coeffs_t[i].to(x.device)
        return result

    return approx_sigmoid

def _make_poly_eval(coeffs):
    coeffs_t = torch.tensor(coeffs, dtype=torch.float32)
    def poly_fn(x):
        result = coeffs_t[-1].to(x.device)
        for i in range(len(coeffs_t) - 2, -1, -1):
            result = result * x + coeffs_t[i].to(x.device)
        return result
    return poly_fn

def _make_approx_coxph(spec):
    parts = spec.split("-")
    degree = int(parts[0])
    exp_k = int(parts[1]) if len(parts) > 1 else 5

    divisor = float(2 ** exp_k)
    bound = lambda c, d: next((float(p[1:]) for p in parts[2:] if p[0] == c), d) / divisor
    exp_coeffs = _remez_fit(np.exp, degree, -bound('l', divisor), bound('u', 0.0))
    approx_exp_base = _make_poly_eval(exp_coeffs)

    def approx_exp(x):
        x_reduced = x / divisor
        result = approx_exp_base(x_reduced)
        for _ in range(exp_k):
            result = result * result
        return result

    if len(parts) > 2:
        opts = {p[0]: float(p[1:]) for p in parts[2:] if p != 'm'}
        approx_exp.shift = 'mean'
        c = _inv_seed(int(opts.get('s', 3)), 10.0 ** -opts['r'])
        seed, B, steps = _make_poly_eval(c), opts.get('b', 1.0), int(opts.get('n', 3))

        def approx_inv_he(x):
            n = B * x.numel()
            t = x / n
            y = seed(t)
            for _ in range(steps):
                y = y * (2 - t * y)
            return y / n
        return approx_exp, approx_inv_he

    inv_coeffs = _remez_fit(lambda x: 1.0 / x, degree, 0.5, 2.0)
    approx_inv_raw = _make_poly_eval(inv_coeffs)

    def approx_inv(x):
        k = torch.floor(torch.log2(x)).long()
        scale = (2.0 ** k.float())
        x_norm = x / scale
        return approx_inv_raw(x_norm) / scale

    return approx_exp, approx_inv

def _inv_seed(degree, lo, n=4001):
    from scipy.optimize import linprog
    t = np.geomspace(lo, 1.0, n)
    A = t[:, None] * t[:, None] ** np.arange(degree + 1)
    ones = np.ones((n, 1))
    res = linprog(np.r_[np.zeros(degree + 1), 1.0],
                  A_ub=np.block([[A, -ones], [-A, -ones]]), b_ub=np.r_[np.ones(n), -np.ones(n)],
                  bounds=[(None, None)] * (degree + 1) + [(0, None)], method='highs')
    return res.x[:degree + 1]
