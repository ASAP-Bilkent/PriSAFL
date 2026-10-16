import numpy as np
from he.prisafltext import PriSAFLText
from he.poly_eval import eval_poly_he

def _remez_relu(degree, interval, n_iter=100, n_test=50000):
    z_test = np.linspace(-1, 1, n_test)
    y_test = np.maximum(z_test * interval, 0)

    k = np.arange(degree + 2)
    z_ref = -np.cos(k * np.pi / (degree + 1))

    coeffs = None
    for _ in range(n_iter):
        A = np.zeros((degree + 2, degree + 2))
        b = np.maximum(z_ref * interval, 0)
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

    if coeffs is None:
        raise RuntimeError("Remez iteration failed to converge")

    result = {}
    for i in range(degree + 1):
        if abs(coeffs[i]) > 1e-12:
            result[i] = float(coeffs[i])
    return result

class Activation:
    APPROX_RELU_POLYS = {
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

    def __init__(self, args):
        self.args = args
        self._approx_relu_scaled_cache = {}

        if args.act_func is None or args.act_func == 'none':
            self.plain_forward = self.none
            self.plain_backward = self.none_derivative
            self.forward = self.none
            self.backward = self.none_derivative
        elif args.act_func == 'approx_relu':
            self.approx_degree = getattr(args, 'approx_degree', 3)
            self.approx_interval = abs(getattr(args, 'approx_b', 1.0)) or 1.0
            self.forward = self.approx_relu
            self.backward = self.approx_relu_derivative
            self.plain_forward = self.approx_relu
            self.plain_backward = self.approx_relu_derivative
        else:
            raise ValueError(f"Activation '{args.act_func}' not supported in HE mode. Use 'approx_relu' or 'none'.")

    def _ensure_no_nan(self, value, func_name):
        if isinstance(value, np.ndarray):
            has_nan = np.isnan(value).any()
        elif np.isscalar(value):
            has_nan = np.isnan(value)
        else:
            return value
        if has_nan:
            raise ValueError(f"NaN detected in {func_name} output")
        return value

    def none(self, x, **_):
        return self._ensure_no_nan(x, 'none')

    def none_derivative(self, z, **_):
        return self._ensure_no_nan(1, 'none_derivative')

    def _eval_poly_ps(self, x, coeffs_dict):
        sorted_powers = sorted([p for p in coeffs_dict.keys() if p > 0], reverse=True)
        if not sorted_powers:
            return coeffs_dict.get(0, 0)

        result = coeffs_dict[sorted_powers[0]]

        for i in range(1, len(sorted_powers)):
            power = sorted_powers[i]
            prev_power = sorted_powers[i-1]
            power_diff = prev_power - power
            x_power_diff = np.power(x, power_diff)
            result = result * x_power_diff + coeffs_dict[power]

        min_power = sorted_powers[-1]
        if min_power > 0:
            x_min_power = np.power(x, min_power)
            result = result * x_min_power

        if 0 in coeffs_dict:
            result = result + coeffs_dict[0]

        return result

    def _eval_poly_derivative_ps(self, x, coeffs_dict):
        derivative_coeffs = {}
        for power, coeff in coeffs_dict.items():
            if power > 0:
                derivative_coeffs[power - 1] = coeff * power
        return self._eval_poly_ps(x, derivative_coeffs)

    def _eval_poly_ps_he(self, x, coeffs_dict):
        return eval_poly_he(x, coeffs_dict)

    def _eval_poly_derivative_ps_he(self, x, coeffs_dict):
        derivative_coeffs = {}
        for power, coeff in coeffs_dict.items():
            if power > 0:
                derivative_coeffs[power - 1] = coeff * power
        return self._eval_poly_ps_he(x, derivative_coeffs)

    def _get_approx_relu_coeffs(self, interval_override=None):
        degree = int(round(self.approx_degree))
        interval = float(interval_override if interval_override is not None else self.approx_interval)
        poly_by_interval = self.APPROX_RELU_POLYS.get(degree, {})
        for candidate in poly_by_interval:
            if np.isclose(candidate, interval):
                return candidate, poly_by_interval[candidate]
        coeffs = _remez_relu(degree, interval)
        self.APPROX_RELU_POLYS.setdefault(degree, {})[interval] = coeffs
        return interval, coeffs

    def _scaled_relu_coeffs(self, interval_key, coeffs):
        cache_key = (int(round(self.approx_degree)), float(interval_key))
        scaled = self._approx_relu_scaled_cache.get(cache_key)
        if scaled is None:
            scaled = {p: c if p == 0 else c / interval_key**p for p, c in coeffs.items()}
            self._approx_relu_scaled_cache[cache_key] = scaled
        return scaled

    def approx_relu(self, x, **_):
        interval_key, coeffs = self._get_approx_relu_coeffs(interval_override=self.approx_interval)
        scaled_coeffs = self._scaled_relu_coeffs(interval_key, coeffs)

        if isinstance(x, PriSAFLText):
            result = self._eval_poly_ps_he(x, scaled_coeffs)
            result.shape = x.shape
            return self._ensure_no_nan(result, 'approx_relu')

        result = self._eval_poly_ps(x, scaled_coeffs)
        return self._ensure_no_nan(result, 'approx_relu')

    def approx_relu_derivative(self, x, **_):
        interval_key, coeffs = self._get_approx_relu_coeffs(interval_override=self.approx_interval)
        scaled_coeffs = self._scaled_relu_coeffs(interval_key, coeffs)

        if isinstance(x, PriSAFLText):
            derivative = self._eval_poly_derivative_ps_he(x, scaled_coeffs)
            derivative.shape = x.shape
            return self._ensure_no_nan(derivative, 'approx_relu_derivative')

        derivative = self._eval_poly_derivative_ps(x, scaled_coeffs)
        return self._ensure_no_nan(derivative, 'approx_relu_derivative')
