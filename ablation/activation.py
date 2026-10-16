import torch
import torch.nn as nn
import numpy as np

def _remez_fit(func, degree, interval_lo, interval_hi, n_iter=100, n_test=50000):
    z_test = np.linspace(interval_lo, interval_hi, n_test)
    y_test = func(z_test)

    k = np.arange(degree + 2)
    z_ref = interval_lo + (interval_hi - interval_lo) * (1 - np.cos(k * np.pi / (degree + 1))) / 2
    coeffs = None
    for _ in range(n_iter):
        A = np.zeros((degree + 2, degree + 2))
        b = func(z_ref)
        for i in range(degree + 2):
            for j in range(degree + 1):
                A[i, j] = z_ref[i] ** j
            A[i, degree + 1] = (-1) ** i
        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        coeffs = sol[:degree + 1]
        p_test = sum(coeffs[j] * z_test**j for j in range(degree + 1))
        abs_err = np.abs(y_test - p_test)
        peaks = [0]
        for i in range(1, n_test - 1):
            if abs_err[i] > abs_err[i-1] and abs_err[i] > abs_err[i+1]:
                peaks.append(i)
        peaks.append(n_test - 1)
        if len(peaks) < degree + 2:
            break
        peak_errs = sorted([(abs_err[p], p) for p in peaks], reverse=True)
        new_refs = sorted([z_test[p] for _, p in peak_errs[:degree + 2]])
        if np.allclose(new_refs, z_ref, atol=1e-12):
            break
        z_ref = np.array(new_refs)
    return coeffs

def _sparse_coeffs(coeffs, degree, tol=1e-12):
    return {i: float(coeffs[i]) for i in range(degree + 1) if abs(coeffs[i]) > tol}

def _remez_relu(degree, interval, n_iter=100, n_test=50000):
    coeffs = _remez_fit(lambda z: np.maximum(z * interval, 0), degree,
                        -1.0, 1.0, n_iter=n_iter, n_test=n_test)
    return _sparse_coeffs(coeffs, degree)

REMEZ_RELU_COEFFS = {
    3: {
        1: {0: 0.0625062499, 1: 0.5, 2: 0.49999},
        3: {0: 0.1875187498, 1: 1.5, 2: 1.49997},
        5: {0: 0.3125312496, 1: 2.5, 2: 2.49995},
        10: {0: 0.6250624992, 1: 5.0, 2: 4.9999},
        15: {0: 0.9375937489, 1: 7.5, 2: 7.49985},
    },
    5: {
        1: {0: 0.0338168537, 1: 0.5, 2: 0.9651121631, 4: -0.5327358706},
        3: {0: 0.101450561, 1: 1.5, 2: 2.8953364892, 4: -1.5982076117},
        5: {0: 0.1690842683, 1: 2.5, 2: 4.8255608153, 4: -2.6636793529},
        10: {0: 0.3381685366, 1: 5.0, 2: 9.6511216307, 4: -5.3273587058},
        15: {0: 0.5072528049, 1: 7.5, 2: 14.476682446, 4: -7.9910380587},
    },
    7: {
        1: {0: 0.0229709768, 1: 0.5, 2: 1.4339290398, 4: -2.0889808107, 6: 1.1550417713},
        3: {0: 0.0689129303, 1: 1.5, 2: 4.3017871195, 4: -6.2669424321, 6: 3.4651253138},
        5: {0: 0.1148548838, 1: 2.5, 2: 7.1696451991, 4: -10.4449040536, 6: 5.7752088563},
        10: {0: 0.2297097676, 1: 5.0, 2: 14.3392903982, 4: -20.8898081071, 6: 11.5504177127},
        15: {0: 0.3445646514, 1: 7.5, 2: 21.5089355973, 4: -31.3347121607, 6: 17.325626569},
    },
    9: {
        1: {0: 0.017351326, 1: 0.5, 2: 1.904757187, 4: -5.1810061204, 6: 6.8588441049, 8: -3.117287824},
        3: {0: 0.052053978, 1: 1.5, 2: 5.7142715609, 4: -15.5430183612, 6: 20.5765323147, 8: -9.3518634721},
        5: {0: 0.08675663, 1: 2.5, 2: 9.5237859348, 4: -25.905030602, 6: 34.2942205245, 8: -15.5864391201},
        10: {0: 0.17351326, 1: 5.0, 2: 19.0475718695, 4: -51.8100612039, 6: 68.5884410489, 8: -31.1728782402},
        15: {0: 0.26026989, 1: 7.5, 2: 28.5713578043, 4: -77.7150918059, 6: 102.8826615734, 8: -46.7593173603},
    },
}

def get_relu_coeffs(degree, interval):
    iv_key = int(interval) if interval == int(interval) else interval
    by_interval = REMEZ_RELU_COEFFS.get(degree)
    if by_interval:
        coeffs = by_interval.get(iv_key)
        if coeffs:
            return coeffs
    return _remez_relu(degree, interval)

class PolyActivationFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeffs, interval):
        ctx.save_for_backward(x)
        ctx.coeffs = coeffs
        ctx.interval = interval
        z = x / interval
        out = torch.zeros_like(x)
        for power, coeff in coeffs.items():
            out = out + coeff * z.pow(power)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        coeffs = ctx.coeffs
        interval = ctx.interval
        z = x / interval
        grad = torch.zeros_like(x)
        for power, coeff in coeffs.items():
            if power == 0:
                continue
            grad = grad + coeff * power * z.pow(power - 1)
        grad = grad / interval
        return grad_output * grad, None, None

class Activation(nn.Module):
    def __init__(self, act_func=None):
        super().__init__()
        if act_func is None:
            self.mode = "relu"
        elif act_func == "sq":
            self.mode = "poly"
            self.coeffs = {2: 1.0}
            self.interval = 1.0
        else:
            parts = act_func.split("-")
            degree = int(parts[0])
            interval = float(parts[1])
            self.mode = "poly"
            self.coeffs = get_relu_coeffs(degree, interval)
            self.interval = interval

    def forward(self, x):
        if self.mode == "relu":
            return torch.relu(x)
        return PolyActivationFunc.apply(x, self.coeffs, self.interval)
