import argparse
import resource
import sys
import time

import numpy as np

from he.main import DIMS, LR, MOMENTUM, APPROX_DEGREE, APPROX_INTERVAL, load_data
from he.encryption import CKKSContext, LEVELS_AFTER_BOOTSTRAP, INTMPBOOT_INPUT_LEVELS
from he.coxcc_loss import pair_events_controls
from he.tests.test_utils import (
    ModelArgs, load_pathway_mask, decrypt_scalar, plaintext_forward,
    plaintext_backward, get_weights, set_weights, sigmoid_plain,
)

L = LEVELS_AFTER_BOOTSTRAP
TOL_W, TOL_S, TOL_MP = 1e-3, 1e-2, 1e-6
SIGMOID_INTERVAL = float(APPROX_INTERVAL)

EXPECTED = {'21': 1, '4': 2, "5'": 1, '5': 2, '6': 2, '9': 0, '10': 2, '12': 0,
            '13': 0, '14': 1, '15': 2, '16': 1, '17-19': 0, '22': 1, '23': 1}
CRITICAL = ['21', '4', '5', '6', '9', '10', '12', '13', '15', '16', '17-19', '22', '23']
W2_SLACK = 2
assert sum(EXPECTED[k] for k in CRITICAL) == L

STEPS = [
    ('21', 'W1', 'look-ahead y1 = W1 + mu(W1 - W1prev) [CD]', r'Look-ahead $\tilde y_1$ ($\mu$, scalar)', ['W1_in'], 'y1'),
    ('21', 'W2', 'look-ahead y2 = W2 + mu(W2 - W2prev) [CD]', r'Look-ahead $\tilde y_2$ ($\mu$, scalar)', ['W2_in'], 'y2'),
    ('4', 'W1', 'Z1 = y1 . (M o P): MultCP + rotate-and-sum + mask', r'$\tilde Z_1$: $\MultCP$, rotate-and-sum, mask', ['y1'], 'Z1'),
    ("5'", 'W1', "phi'(Z1) (kept for line 15)", r"$\varphi'(\tilde Z_1)$ (parallel)", ['Z1'], 'dphi1'),
    ('5', 'W1', 'A1 = phi(Z1)', r'$\tilde A_1 = \varphi(\tilde Z_1)$', ['Z1'], 'A1'),
    ('6', 'W1+W2', 'h = y2 . A1: MultCC + rotate-and-sum + mask', r'$\tilde h$: $\MultCC$, rotate-and-sum, mask', ['A1', 'y2'], 'h'),
    ('9', 'W1+W2', 'delta_ij = h_j - h_i', r'$\tilde\delta_{ij} = \tilde h^{(j)} - \tilde h^{(i)}$', ['h'], 'delta'),
    ('10', 'W1+W2', 'sigma_poly(delta_ij)', r'$\sigma_{\mathrm{poly}}(\tilde\delta_{ij})$', ['delta'], 'sigma'),
    ('12', 'W1+W2', 'accumulate +/- g', r'Accumulate $\pm\tilde g$', ['sigma'], 'g'),
    ('13', 'W1+W2', 'delta2 = g (right rotate-replicate)', r'$\tilde\delta_2 \gets \tilde g$ (rotate-replicate)', ['g'], 'delta2'),
    ('14', 'W2', 'dW2 = A1 o delta2 [CC] (parallel)', r'$\nabla\tilde W_2 = \tilde A_1 \odot \tilde\delta_2$ (parallel)', ['A1', 'delta2'], 'dW2'),
    ('15', 'W1', "delta1 = (delta2 . y2^T) o phi'(Z1) [CC, CC]", r"$\tilde\delta_1 = \tilde\delta_2 \tilde W_2^T \odot \varphi'(\tilde Z_1)$", ['delta2', 'y2', 'dphi1'], 'delta1'),
    ('16', 'W1', 'dW1 = M^T o delta1 [CC, M^T encrypted]', r'$\nabla\tilde W_1 = \hat M^T \odot \tilde\delta_1$', ['delta1'], 'dW1'),
    ('17-19', 'W1', 'sum over samples and clients', r'Aggregate $\nabla\tilde W_1$ (additions)', ['dW1'], 'dW1_agg'),
    ('17-19', 'W2', 'sum over samples and clients', r'Aggregate $\nabla\tilde W_2$ (additions)', ['dW2'], 'dW2_agg'),
    ('22', 'W1', 'W1 = y1 - (eta/N) dW1 [CD]', r'$\tilde W_1 = \tilde y_1 - \eta\,\Delta\tilde W_1$', ['y1', 'dW1_agg'], 'W1_new'),
    ('22', 'W2', 'W2 = y2 - (eta/N) dW2 [CD]', r'$\tilde W_2 = \tilde y_2 - \eta\,\Delta\tilde W_2$', ['y2', 'dW2_agg'], 'W2_new'),
    ('23', 'W1', 'Bootstrap(W1)', r'Bootstrap($\tilde W_1$)', ['W1_new'], 'W1_boot'),
    ('23', 'W2', 'Bootstrap(W2)', r'Bootstrap($\tilde W_2$)', ['W2_new'], 'W2_boot'),
]

REC = {}
BOOT = {}
NEG = {}
STATE = {'on': False, 'neg': False, 'rank': None, 'z': None, 'deriv': None}


def eff(ht):
    ct = ht._cipher_data
    return 0 if ct is None else ct.GetLevel() + ct.GetNoiseScaleDeg() - 1


def rec(node, ht):
    if STATE['on'] and ht is not None:
        REC.setdefault(node, []).append(eff(ht))


def decrypt_valid(ht, ct=None):
    from he.prisafltext import PriSAFLText
    pt = PriSAFLText._decrypt(ct if ct is not None else ht._cipher_data)
    pt.SetLength(ht.valid_len)
    return np.array(pt.GetRealPackedValue())[ht.validation_mask == 1]


def negative_control(ht, before, boot):
    from he.prisafltext import PriSAFLText
    ct = PriSAFLText.ckks_context.cc.EvalMult(ht._cipher_data, 1.0)
    NEG['eff'] = ct.GetLevel() + ct.GetNoiseScaleDeg() - 1
    tmp = PriSAFLText(ct, mask=ht.validation_mask)
    try:
        boot(tmp)
        NEG['err'] = float(np.max(np.abs(decrypt_valid(tmp) - before)))
    except Exception as ex:
        NEG['raised'] = ' '.join(str(ex).split())[-160:]


def install_hooks():
    import he.trainer as trainer
    import he.coxcc_loss as coxcc
    from he.layer import LayerHE
    from he.model import PriSAFLHE
    from he.prisafltext import PriSAFLText

    labels = {'after matmul (PC)': 'Z1', 'after matmul (CC)': 'h',
              'after act_deriv': 'dphi1', 'after activation': 'A1',
              'after W update': 'W{r}_new', 'after W bootstrap': 'W{r}_boot'}

    def log_level(self, label, ht):
        if label in labels:
            rec(labels[label].format(r=self.rank), ht)
        if label.startswith('after matmul') and self.W._cipher_data is not None:
            rec(f'W{self.rank}_fwd', self.W)
    LayerHE._log_level = log_level

    orig_la = PriSAFLHE.apply_lookahead
    def apply_lookahead(self):
        for layer in self.layers:
            rec(f'W{layer.rank}_in', layer.W)
        orig_la(self)
        for layer in self.layers:
            rec(f'y{layer.rank}', layer.W)
    PriSAFLHE.apply_lookahead = apply_lookahead

    orig_sig = coxcc.sigmoid_he
    def sigmoid_he(x, **kw):
        rec('delta', x)
        out = orig_sig(x, **kw)
        rec('sigma', out)
        return out
    coxcc.sigmoid_he = sigmoid_he

    orig_grad = trainer.compute_coxcc_gradient
    def compute_coxcc_gradient(*a, **kw):
        out = orig_grad(*a, **kw)
        for g in out:
            rec('g', g)
        return out
    trainer.compute_coxcc_gradient = compute_coxcc_gradient

    orig_bw = LayerHE.backward
    def backward(self, z, g, bs):
        STATE.update(rank=self.rank, z=z, deriv=None if self.last else self.preZ_deriv)
        gW, zg = orig_bw(self, z, g, bs)
        STATE.update(rank=None, z=None, deriv=None)
        rec(f'dW{self.rank}', zg)
        return gW, zg
    LayerHE.backward = backward

    orig_mult = PriSAFLText.mult
    def mult(self, other, mult_types="CC"):
        out = orig_mult(self, other, mult_types)
        if STATE['z'] is not None and self is STATE['z'] and STATE['rank'] == 2:
            rec('delta2', other)
        if STATE['deriv'] is not None and other is STATE['deriv']:
            rec('delta1', out)
        return out
    PriSAFLText.mult = mult

    orig_uw = PriSAFLHE.update_weights
    def update_weights(self, grads, *a, **kw):
        for i, g in enumerate(grads):
            rec(f'dW{i + 1}_agg', g)
        return orig_uw(self, grads, *a, **kw)
    PriSAFLHE.update_weights = update_weights

    orig_luw = LayerHE.update_weights
    def layer_update_weights(self, *a, **kw):
        STATE['rank'] = self.rank
        return orig_luw(self, *a, **kw)
    LayerHE.update_weights = layer_update_weights

    orig_boot = PriSAFLText.apply_bootstrapping
    def apply_bootstrapping(self):
        r = STATE['rank']
        before = decrypt_valid(self)
        if STATE['neg'] and r == 1:
            negative_control(self, before, orig_boot)
        orig_boot(self)
        if STATE['on']:
            BOOT[r] = float(np.max(np.abs(decrypt_valid(self) - before)))
    PriSAFLText.apply_bootstrapping = apply_bootstrapping


def score(model, x):
    s, _ = plaintext_forward(model, x)
    return float(np.ravel(s)[0])


def lookahead(model_plain, x, x_prev):
    if model_plain.momentum > 0.0 and x_prev is not None:
        return [x[i] + model_plain.momentum * (x[i] - x_prev[i]) for i in range(len(x))]
    return x


def plain_step(model_plain, X, parts, lr, x_prev, seed, heavy_ball=False):
    n = len(X)
    x = get_weights(model_plain)
    y = lookahead(model_plain, x, x_prev)
    set_weights(model_plain, x if heavy_ball else y)
    scores = np.array([score(model_plain, X[s:s + 1]) for s in range(n)])
    np.random.seed(seed)
    g = np.zeros(n)
    for off, dur, evt in parts:
        for i, j in pair_events_controls(dur, evt):
            sig = sigmoid_plain(scores[off + j] - scores[off + i], SIGMOID_INTERVAL, APPROX_DEGREE)
            g[off + j] += sig
            g[off + i] -= sig
    acc = None
    for s in np.nonzero(g)[0]:
        _, inter = plaintext_forward(model_plain, X[s:s + 1])
        sg = plaintext_backward(model_plain, inter, np.full((1, 1), g[s]))
        acc = sg if acc is None else [a + b for a, b in zip(acc, sg)]
    set_weights(model_plain, [y[i] - lr / n * acc[i] for i in range(len(y))])
    return x


def packed(layer, W):
    from he.prisafltext import PriSAFLText
    P = PriSAFLText(np.pad(W, ((0, layer.padded_in_dim - layer.in_dim),
                               (0, layer.padded_out_dim - layer.out_dim))))
    odd = layer.rank % 2 == 1
    step = layer.slot_multipliers[layer.rank - 1] * (layer.padded_in_dim if odd else layer.padded_out_dim)
    P._pad_data('col' if odd else 'row', pad_step=step, pad_batch=1)
    return P.padded_data


def mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable'):
                return int(line.split()[1]) / 2**20
    return float('inf')


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def main():
    p = argparse.ArgumentParser(description="Level/budget test, 2-layer CoxCC (Alg. 1)")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--n-iters", type=int, default=4,
                   help="iteration 1 is epoch 1, 2 is the transition, >=3 steady (single-party)")
    p.add_argument("--n-clients", type=int, default=1)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--momentum", type=float, default=MOMENTUM)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rss-gb", type=float, default=35.0,
                   help="skip --n-clients > 1 if less memory than this is available")
    args = p.parse_args()
    multi = args.n_clients > 1
    first_steady = 2 if multi else 3
    if args.n_iters < first_steady + 1:
        p.error(f"--n-iters must be >= {first_steady + 1} (two steady-state iterations)")
    if multi and mem_available_gb() < args.max_rss_gb:
        print(f"SKIP: --n-clients {args.n_clients} needs up to ~{args.max_rss_gb:.0f} GB, "
              f"only {mem_available_gb():.1f} GB available")
        sys.exit(0)

    from sklearn.preprocessing import StandardScaler
    from he.prisafltext import PriSAFLText
    from he.model import PriSAFLHE
    from he.trainer import train_epoch_he

    np.random.seed(args.seed)
    quant, events, survival, _ = load_data()
    n = args.n_samples
    X = StandardScaler().fit(quant[:n]).transform(quant[:n])
    dur, evt = survival[:n], events[:n]
    mask = load_pathway_mask()

    t0 = time.time()
    ctx = CKKSContext(n_clients=args.n_clients)
    PriSAFLText.set_ckks_context(ctx)
    depth = ctx.depth
    refresh = L - 1 + INTMPBOOT_INPUT_LEVELS if multi else L
    boot_min = INTMPBOOT_INPUT_LEVELS if multi else 1
    tol_w, tol_s = (TOL_MP, TOL_MP) if multi else (TOL_W, TOL_S)
    print(f"Setup {time.time() - t0:.0f}s: ring 2^{ctx.cc.GetRingDimension().bit_length() - 1}, "
          f"depth {depth}, L {L}, clients {args.n_clients}, samples {n}, events {int(evt.sum())}")

    margs = ModelArgs(momentum=args.momentum)
    model = PriSAFLHE(DIMS, mask, margs)
    model_plain = PriSAFLHE(DIMS, mask, margs)
    model_hb = PriSAFLHE(DIMS, mask, margs)
    set_weights(model_plain, [w.copy() for w in get_weights(model)])
    set_weights(model_hb, [w.copy() for w in get_weights(model)])

    client_data, parts = None, [(0, dur, evt)]
    if multi:
        piece = n // args.n_clients
        bounds = [(c * piece, (c + 1) * piece if c < args.n_clients - 1 else n)
                  for c in range(args.n_clients)]
        client_data = [(X[a:b], dur[a:b], evt[a:b]) for a, b in bounds]
        parts = [(a, dur[a:b], evt[a:b]) for a, b in bounds]

    install_hooks()
    checks, iters, x_prev, x_prev_hb = [], [], None, None

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    for it in range(1, args.n_iters + 1):
        REC.clear()
        BOOT.clear()
        STATE.update(on=True, neg=(not multi and it == args.n_iters))
        seed = args.seed + 1000 + it
        t1 = time.time()
        np.random.seed(seed)
        try:
            train_epoch_he(model, X, dur, evt, args.lr, n_clients=args.n_clients,
                           client_data=client_data, bootstrap=True,
                           sigmoid_interval=SIGMOID_INTERVAL, sigmoid_degree=APPROX_DEGREE)
            raised = None
        except Exception as ex:
            raised = ' '.join(str(ex).split())[-200:]
        STATE.update(on=False, neg=False)
        check(f"iter {it}: no HE op raises", raised is None, raised or "")
        if raised:
            break
        nonuni = {k: sorted(set(v)) for k, v in REC.items() if len(set(v)) > 1}
        iters.append(({k: max(v) for k, v in REC.items()}, dict(BOOT), nonuni))
        x_prev = plain_step(model_plain, X, parts, args.lr, x_prev, seed)
        x_prev_hb = plain_step(model_hb, X, parts, args.lr, x_prev_hb, seed, heavy_ball=True)
        print(f"iter {it}: {time.time() - t1:.0f}s, peak RSS {peak_rss_gb():.1f} GB"
              + (f", per-sample level spread {nonuni}" if nonuni else ""), flush=True)

    def rows(it, e):
        out = []
        for line, path, op, tex, ins, node in STEPS:
            before, after = max(e[i] for i in ins), e[node]
            crit = line in CRITICAL and path != 'W2'
            if line == '23':
                arrival = depth - before
                refreshed = not multi and after != before
                d = EXPECTED['23'] if refreshed else None
                exp = EXPECTED['23'] if refreshed and it >= first_steady else None
                ok = None if exp is None else arrival == 1 + (W2_SLACK if path == 'W2' else 0)
            else:
                d = after - before
                exp = 0 if (line == '21' and it == 1) else EXPECTED[line]
                ok = d == exp
            out.append((line, path, crit, op, tex, before, after, d, exp, ok))
        return out

    def consumed_str(row, tex=False):
        line, path, _, _, _, before, _, d, _, _ = row
        if d is None:
            return '--' if tex else ('- (IntMPBoot)' if multi else '- (no-op)')
        if line != '23':
            return str(d)
        unused = depth - before - d
        return f"{d} (headroom" + (f", +{unused} unused)" if unused else ")")

    tables = []
    for it, (e, boot, _) in enumerate(iters, 1):
        r = rows(it, e)
        tables.append(r)
        for line, path, _, _, _, b, a, d, exp, ok in r:
            if ok is not None:
                what = "remaining on arrival" if line == '23' else "delta"
                check(f"iter {it}: line {line} {path} {what}", ok,
                      f"{depth - b}" if line == '23' else f"{d} vs {exp}")
        if it >= first_steady:
            total = sum(row[7] or 0 for row in r if row[2])
            want = L if not multi else L - 1
            check(f"iter {it}: critical-path Consumed column sums to {want}"
                  + ("" if multi else f" ({L - 1} + 1 headroom)"), total == want, f"{total}")

    rem = lambda e, k: depth - e[k]
    for it, (e, boot, _) in enumerate(iters, 1):
        for r in (1, 2):
            check(f"iter {it}: W{r} reaches bootstrap with >= {boot_min} remaining",
                  rem(e, f'W{r}_new') >= boot_min, f"{rem(e, f'W{r}_new')}")
            check(f"iter {it}: W{r} bootstrap decrypts (|after-before| < {tol_w:g})",
                  boot.get(r, np.inf) < tol_w, f"{boot.get(r, np.inf):.2e}")
        if it == 1:
            fresh = refresh if multi else depth
            check(f"iter 1: first forward encrypts W1, W2 fresh (full chain, {fresh})",
                  all(f'W{r}_fwd' in e and rem(e, f'W{r}_fwd') == fresh for r in (1, 2)),
                  f"{[rem(e, f'W{r}_fwd') if f'W{r}_fwd' in e else None for r in (1, 2)]}")
            if not multi:
                check("iter 1: EvalBootstrap is a level no-op (input above L)",
                      e['W1_boot'] == e['W1_new'], f"{rem(e, 'W1_new')} -> {rem(e, 'W1_boot')}")
        if multi or it >= 2:
            for r in (1, 2):
                check(f"iter {it}: W{r} refreshed to {refresh} remaining",
                      rem(e, f'W{r}_boot') == refresh, f"{rem(e, f'W{r}_boot')}")
        if it >= first_steady:
            consumed = e['W1_new'] - e['W1_in']
            check(f"iter {it}: W1, W2 enter with {refresh} remaining",
                  rem(e, 'W1_in') == refresh and rem(e, 'W2_in') == refresh,
                  f"{rem(e, 'W1_in')}, {rem(e, 'W2_in')}")
            check(f"iter {it}: W1 path consumes {L - 1}", consumed == L - 1, f"{consumed}")
    if not multi and iters:
        raised = NEG.get('raised', '')
        ok = NEG.get('eff') == depth and ('DropLastElement' in raised or NEG.get('err', 0) > tol_w)
        check("negative control: 1 extra mult (eff == depth) makes the production bootstrap fail",
              ok, f"eff {NEG.get('eff')} (remaining {depth - NEG.get('eff', 0)}): "
              + (f"raised '{raised}'" if raised else f"err {NEG.get('err')}"))

    w_err, hb_err = [], []
    for layer, W, Whb in zip(model.layers, get_weights(model_plain), get_weights(model_hb)):
        dec = np.array(layer.W._return_decrypted())
        ref, ref_hb = packed(layer, W), packed(layer, Whb)
        m = min(len(dec), len(ref))
        sel = layer.W.validation_mask[:m] == 1
        if layer._forward_mask_packed is not None:
            off = sel & (layer._forward_mask_packed[:m] == 0)
            sel = sel & (layer._forward_mask_packed[:m] == 1)
            print(f"W{layer.rank}: {off.sum()} off-pathway slots drift up to "
                  f"{np.max(np.abs(dec[:m][off])):.2e} (masked out of every forward)")
        err = float(np.max(np.abs(dec[:m][sel] - ref[:m][sel])))
        w_err.append(err)
        hb_err.append(float(np.max(np.abs(dec[:m][sel] - ref_hb[:m][sel]))))
        check(f"W{layer.rank} HE vs plaintext (max abs, {sel.sum()} slots)", err < tol_w, f"{err:.2e} < {tol_w:g}")
    s_err = max(abs(decrypt_scalar(model.forward(X[s:s + 1])[-1]) - score(model_plain, X[s:s + 1]))
                for s in range(n))
    s_max = max(abs(score(model_plain, X[s:s + 1])) for s in range(n))
    check("risk scores HE vs plaintext (max abs)", s_err < tol_s,
          f"{s_err:.2e} < {tol_s:g} (rel {s_err / s_max:.1e} of max|S| {s_max:.2e})")
    order = (f"HE vs Nesterov mirror {max(w_err):.2e}, vs heavy-ball mirror {max(hb_err):.2e}")
    if multi:
        check("gradient at look-ahead y: err(Nesterov) < tol < err(heavy-ball)",
              max(w_err) < tol_w < max(hb_err), order)
    else:
        print(f"NOTE: look-ahead order not checked numerically ({order}); EvalBootstrap noise "
              f"hides the Nesterov/heavy-ball gap, the order is established by the line-21 level trace.")

    print(f"\nLevel convention: eff = GetLevel()+GetNoiseScaleDeg()-1 (consumed); "
          f"rem = depth({depth}) - eff. Cells are eff/rem. crit '*' = W1 critical path.")
    for it, r in enumerate(tables, 1):
        tag = "steady state" if it >= first_steady else ("epoch 1" if it == 1 else "transition")
        print(f"\n### Iteration {it} ({tag})\n")
        print("| Alg.1 line | path | crit | op | level before | level after | consumed | expected | pass |")
        print("|---|---|---|---|---|---|---|---|---|")
        for row in r:
            line, path, crit, op, _, b, a, d, exp, ok = row
            print(f"| {line} | {path} | {'*' if crit else ''} | {op} | {b}/{depth - b} | {a}/{depth - a} | "
                  f"{consumed_str(row)} | {'-' if exp is None else exp} | "
                  f"{'-' if ok is None else ('OK' if ok else 'FAIL')} |")

    if tables:
        it = len(tables)
        print(f"\n% LaTeX tabular body, iteration {it}, measured, W1 critical path only "
              f"(remaining levels, depth {depth})")
        print(r"% Alg.1 line & Operation & Remaining before & Remaining after & Consumed & Expected \\")
        for row in tables[-1]:
            line, path, crit, op, tex, b, a, d, exp, ok = row
            if crit:
                print(f"{line} & {tex} & {depth - b} & {depth - a} & {consumed_str(row, tex=True)} & "
                      f"{'--' if exp is None else exp} \\\\")
        print("% Off the critical path (parallel or W2; not summed):")
        for row in tables[-1]:
            line, path, crit, op, tex, b, a, d, exp, ok = row
            if not crit:
                print(f"%   {line} {path}: {op}: rem {depth - b} -> {depth - a}, consumed {consumed_str(row)}")

    print("\n## Checks")
    for name, ok, detail in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    passed = bool(checks) and all(ok for _, ok, _ in checks)
    print(f"\nW max abs err: {', '.join(f'{x:.2e}' for x in w_err)} (tol {tol_w:g}); "
          f"score max abs err: {s_err:.2e} (tol {tol_s:g})")
    print(f"Peak RSS: {peak_rss_gb():.1f} GB; wall time {time.time() - t0:.0f}s")
    print(f"\n{'=' * 60}\nRESULT: {'PASSED' if passed else 'FAILED'}\n{'=' * 60}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
