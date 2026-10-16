import numpy as np
from he.prisafltext import PriSAFLText

def make_const_like(value, template):
    n = len(template.validation_mask)
    ht = PriSAFLText(np.array([[value]]))
    ht._padded_data = np.full(n, value) * template.validation_mask
    ht.validation_mask = template.validation_mask.copy()
    ht.shape = template.shape
    return ht

def _squaring_table(x, up_to):
    table = {1: x}
    step = 1
    while step * 2 <= up_to:
        table[step * 2] = table[step].mult(table[step], "CC")
        step *= 2
    return table

def _power(table, n):
    result = None
    for step in sorted(table, reverse=True):
        if step <= n:
            n -= step
            result = table[step] if result is None else result.mult(table[step], "CC")
        if n == 0:
            break
    return result

def eval_poly_he(x, coeffs_dict):
    max_power = max((p for p in coeffs_dict), default=0)
    if max_power == 0:
        return make_const_like(coeffs_dict.get(0, 0.0), x)

    k = int(np.ceil(np.sqrt(max_power + 1)))
    n_chunks = max_power // k + 1
    table = _squaring_table(x, max_power)
    giant = _power(table, k) if n_chunks > 1 else None

    def chunk(base):
        ct = None
        for i in range(1, k):
            c = coeffs_dict.get(base + i, 0.0)
            if c == 0.0:
                continue
            term = _power(table, i).mult(make_const_like(c, x), "CP")
            ct = term if ct is None else ct.add(term, "CC")
        return ct, coeffs_dict.get(base, 0.0)

    acc_ct = None
    acc_const = 0.0
    for j in range(n_chunks - 1, -1, -1):
        ct, const = chunk(j * k)
        if j == n_chunks - 1:
            acc_ct, acc_const = ct, const
            continue
        folded = acc_ct.mult(giant, "CC") if acc_ct is not None else None
        if acc_const != 0.0:
            scaled = giant.mult(make_const_like(acc_const, x), "CP")
            folded = scaled if folded is None else folded.add(scaled, "CC")
        if ct is not None:
            folded = ct if folded is None else folded.add(ct, "CC")
        acc_ct, acc_const = folded, const

    result = acc_ct
    if acc_const != 0.0:
        c_ht = make_const_like(acc_const, x)
        result = c_ht if result is None else result.add(c_ht, "CP")
    if result is None:
        result = make_const_like(0.0, x)
    return result
