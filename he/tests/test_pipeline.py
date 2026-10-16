import argparse
import numpy as np
import sys
import time

from he.main import DIMS as DEFAULT_DIMS, APPROX_DEGREE, APPROX_INTERVAL, LR, MOMENTUM
from he.encryption import CKKSContext
from he.tests.test_utils import (
    ModelArgs, setup_ckks_context, load_pathway_mask,
    decrypt_scalar, plaintext_backward, plaintext_forward,
    get_weights, set_weights, sigmoid_plain,
)
from he.coxcc_loss import compute_coxcc_gradient as _compute_coxcc_gradient, pair_events_controls

SIGMOID_POLY_DEGREE = APPROX_DEGREE
SIGMOID_INTERVAL = float(APPROX_INTERVAL)
BACKWARD_THRESHOLD = 1e-3

STAGES = ['forward', 'coxcc_grad', 'backward', 'update', 'multi_iter']

def _stage_idx(name):
    return STAGES.index(name)

def check(label, he_val, plain_val, threshold=1e-3):
    diff = abs(he_val - plain_val)
    ok = diff < threshold
    status = "OK" if ok else "FAIL"
    print(f"  {label}: HE={he_val:+.8f}, Plain={plain_val:+.8f}, "
          f"diff={diff:.2e} [{status}]")
    return ok

def estimate_levels(n_layers, approx_degree=APPROX_DEGREE,
                    sigmoid_poly_degree=SIGMOID_POLY_DEGREE):
    act_mults = approx_degree - 1
    n_act_layers = max(n_layers - 1, 1)
    fwd = (2 + act_mults) * n_act_layers + 2
    loss_d = sigmoid_poly_degree
    bwd = 4 * n_layers
    return fwd + loss_d + bwd + 15

def print_result(passed):
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASSED' if passed else 'FAILED'}")
    print(f"{'='*60}")
    return passed

def prepare_gradient_for_backward(grad_ct, z_last):
    grad_ct.validation_mask = z_last.validation_mask.copy()
    grad_ct.shape = z_last.shape
    return grad_ct

def compute_coxcc_gradient(encrypted_outputs, events, durations,
                             sigmoid_poly_degree, sigmoid_interval, seed):
    np.random.seed(seed)
    pairs = pair_events_controls(durations, events)
    grads = _compute_coxcc_gradient(
        encrypted_outputs, pairs,
        sigmoid_interval=sigmoid_interval, sigmoid_degree=sigmoid_poly_degree,
    )
    return grads, pairs

def run_pipeline(args):
    np.random.seed(args.seed)

    dims = args.dims
    n_layers = len(dims) - 1
    n_samples = args.n_samples
    n_clients = args.n_clients
    lr = args.lr
    stage = args.stage
    do_bootstrap = args.bootstrap

    target_idx = len(STAGES) - 1 if stage == 'all' else _stage_idx(stage)
    validate_all = (stage == 'all')

    pathway_mask = load_pathway_mask()
    print(f"Pathway mask: {pathway_mask.shape}, density: {pathway_mask.mean():.3f}")

    X = np.random.randn(n_samples, dims[0])
    survival_times = np.random.uniform(30, 1000, n_samples)
    events = np.random.binomial(1, 0.7, n_samples).astype(np.float64)
    sort_idx = np.argsort(-survival_times)
    X = X[sort_idx]
    survival_times = survival_times[sort_idx]
    events = events[sort_idx]
    if events.sum() == 0:
        events[0] = 1.0

    print(f"\nPatients (sorted by descending survival):")
    for i in range(n_samples):
        print(f"  {i}: survival={survival_times[i]:.0f}, event={events[i]:.0f}")

    min_levels = estimate_levels(n_layers,
                                 approx_degree=args.approx_degree,
                                 sigmoid_poly_degree=args.sigmoid_poly_degree)
    dims_str = ' -> '.join(map(str, dims))
    print(f"\nArchitecture: {dims_str}")
    print(f"Layers: {n_layers}, Samples: {n_samples}, Clients: {n_clients}")

    from he.prisafltext import PriSAFLText
    from he.model import PriSAFLHE
    margs = ModelArgs(approx_degree=args.approx_degree, momentum=args.momentum, verbosity=args.verbose)

    def build_context():
        if do_bootstrap:
            print(f"Setting up CKKS+FHE (n_clients={n_clients})...")
            t0 = time.time()
            c = CKKSContext(n_clients=n_clients,
                            ring_dim_exp=args.ring_dim_exp)
            print(f"Setup: {time.time()-t0:.1f}s (total_depth={c.depth})")
        else:
            print(f"Est. levels: {min_levels}, Using: {min_levels}")
            print(f"Setting up CKKS (ring_dim=1<<{args.ring_dim_exp or 15}, levels={min_levels})...")
            t0 = time.time()
            c = setup_ckks_context(args.ring_dim_exp or 15, min_levels)
            print(f"CKKS setup: {time.time()-t0:.1f}s")
        return c

    n_probe = min(n_samples, 5)
    rng_state = np.random.get_state()
    for attempt in range(16):
        np.random.set_state(rng_state)
        ctx = build_context()
        PriSAFLText.set_ckks_context(ctx)
        PriSAFLText.set_verbosity(args.verbose)
        model = PriSAFLHE(dims, pathway_mask, margs)
        good = True
        for s in range(n_probe):
            try:
                if not np.isfinite(decrypt_scalar(model.forward(X[s:s+1])[-1])):
                    good = False
                    break
            except Exception:
                good = False
                break
        if good:
            break
        print(f"  [context probe failed, rebuilding ({attempt + 1})]")
    else:
        raise RuntimeError("Could not build a numerically valid CKKS context")

    initial_weights = get_weights(model)
    model_plain = PriSAFLHE(dims, pathway_mask, margs)
    set_weights(model_plain, [w.copy() for w in initial_weights])

    client_models = [model]
    for c in range(1, n_clients):
        cm = PriSAFLHE(dims, pathway_mask, margs)
        set_weights(cm, [w.copy() for w in initial_weights])
        client_models.append(cm)

    piece_size = n_samples // n_clients
    client_ranges = []
    for c in range(n_clients):
        start = c * piece_size
        end = (c + 1) * piece_size if c < n_clients - 1 else n_samples
        client_ranges.append((start, end))

    if n_clients > 1:
        print(f"\nClient data partitioning:")
        for c in range(n_clients):
            print(f"  Client {c}: samples {client_ranges[c][0]}-{client_ranges[c][1]-1}")

    all_ok = True

    plain_scores_list = []
    for s in range(n_samples):
        scores, _ = plaintext_forward(model_plain, X[s:s+1])
        val = scores.item() if hasattr(scores, 'item') else float(scores[0])
        plain_scores_list.append(val)
    plain_scores = np.array(plain_scores_list)

    print(f"\n--- Phase 1: Forward ({n_samples} samples, {n_clients} clients) ---")
    z_he_all = [None] * n_samples
    derivs_all = [None] * n_samples
    for c in range(n_clients):
        for s in range(client_ranges[c][0], client_ranges[c][1]):
            t0 = time.time()
            z_he = client_models[c].forward(X[s:s+1])
            print(f"  Client {c}, Sample {s}: {time.time()-t0:.1f}s")
            z_he_all[s] = z_he
            derivs_all[s] = [layer.preZ_deriv for layer in client_models[c].layers]

    if validate_all or stage == 'forward':
        print(f"\n  Validate: forward (final layer, decrypt)")
        for s in range(n_samples):
            he_score = decrypt_scalar(z_he_all[s][-1])
            ok = check(f"score[{s}]", he_score, plain_scores[s], threshold=1e-6)
            if not ok:
                all_ok = False
        if stage == 'forward':
            return print_result(all_ok)

    if target_idx < _stage_idx('coxcc_grad'):
        return print_result(all_ok)

    encrypted_outputs = [z_he_all[s][-1] for s in range(n_samples)]

    coxcc_seed = args.seed + 1000
    print(f"\n--- Phase 2: Cox-CC gradient ({n_samples} samples) ---")
    encrypted_coxcc_grads, case_control_pairs = compute_coxcc_gradient(
        encrypted_outputs, events, survival_times,
        args.sigmoid_poly_degree, SIGMOID_INTERVAL, coxcc_seed)

    plain_coxcc_grads = np.zeros(n_samples)
    for i_evt, j_ctrl in case_control_pairs:
        diff = plain_scores[j_ctrl] - plain_scores[i_evt]
        sig = sigmoid_plain(diff, SIGMOID_INTERVAL, args.sigmoid_poly_degree)
        plain_coxcc_grads[j_ctrl] += sig
        plain_coxcc_grads[i_evt] -= sig

    if validate_all or stage == 'coxcc_grad':
        print(f"\n  Validate: coxcc_grad")
        for s in range(n_samples):
            if encrypted_coxcc_grads[s] is None:
                he_grad = 0.0
            else:
                he_grad = decrypt_scalar(encrypted_coxcc_grads[s])
            ok = check(f"coxcc_grad[{s}]", he_grad, plain_coxcc_grads[s],
                       threshold=1e-4)
            if not ok:
                all_ok = False
        if stage == 'coxcc_grad':
            return print_result(all_ok)

    if target_idx < _stage_idx('backward'):
        return print_result(all_ok)

    print(f"\n--- Phase 3: Backward ---")

    client_weight_grads = [([None] * n_layers) for _ in range(n_clients)]
    per_sample_he_grads = {}
    per_sample_plain_grads = {}
    for c in range(n_clients):
        for s in range(client_ranges[c][0], client_ranges[c][1]):
            if encrypted_coxcc_grads[s] is None:
                continue

            for layer, deriv in zip(client_models[c].layers, derivs_all[s]):
                layer.preZ_deriv = deriv

            z_he = z_he_all[s]
            z_last = z_he[-1]

            grad_masked = prepare_gradient_for_backward(
                encrypted_coxcc_grads[s], z_last)
            z_he[-1] = grad_masked

            sample_grads = client_models[c].backward(X[s:s+1], z_he)
            per_sample_he_grads[s] = sample_grads

            grad_scalar_s = decrypt_scalar(encrypted_coxcc_grads[s])
            _, intermediates_s = plaintext_forward(model_plain, X[s:s+1])
            grad_np_s = np.ones((1, dims[-1])) * grad_scalar_s
            per_sample_plain_grads[s] = plaintext_backward(
                model_plain, intermediates_s, grad_np_s, apply_mask=False)

            for i in range(n_layers):
                if client_weight_grads[c][i] is None:
                    client_weight_grads[c][i] = sample_grads[i]
                else:
                    client_weight_grads[c][i] = client_weight_grads[c][i].add(
                        sample_grads[i])

    if n_clients > 1:
        he_weight_grads = [
            PriSAFLText.sum_ciphertexts([client_weight_grads[c][i] for c in range(n_clients)
                            if client_weight_grads[c][i] is not None])
            if any(client_weight_grads[c][i] is not None for c in range(n_clients))
            else None
            for i in range(n_layers)
        ]
    else:
        he_weight_grads = client_weight_grads[0]

    if validate_all or stage == 'backward':
        print(f"\n  Validate: backward (per-sample, numeric)")
        if not all(he_weight_grads[i] is not None for i in range(n_layers)):
            all_ok = False

        for s in sorted(per_sample_he_grads):
            for i in range(n_layers):
                he_g = per_sample_he_grads[s][i]
                if he_g is None:
                    continue
                dec = np.array(he_g._return_decrypted())
                valid = dec[he_g.validation_mask[:len(dec)] == 1]
                he_sorted = np.sort(np.abs(valid))
                ref_sorted = np.sort(np.abs(per_sample_plain_grads[s][i].ravel()))
                if len(he_sorted) != len(ref_sorted):
                    diff = float('inf')
                else:
                    diff = float(np.max(np.abs(he_sorted - ref_sorted)))
                ok = diff < BACKWARD_THRESHOLD
                if not ok:
                    all_ok = False
                print(f"    sample[{s}] grad[L{i}] |sorted| maxdiff={diff:.3e} "
                      f"(thr {BACKWARD_THRESHOLD:.0e}) [{'OK' if ok else 'FAIL'}]")

        if stage == 'backward':
            return print_result(all_ok)

    if target_idx < _stage_idx('update'):
        return print_result(all_ok)

    print(f"\n--- Phase 4: Weight update ---")

    if he_weight_grads[0] is None:
        print(f"  SKIP: no weight gradients")
        return print_result(all_ok)

    for c in range(n_clients):
        client_models[c].update_weights(he_weight_grads, lr / n_samples,
                                         is_last_iteration=False,
                                         bootstrap=n_clients > 1)

    he_grad_scalars = []
    for s in range(n_samples):
        if encrypted_coxcc_grads[s] is None:
            he_grad_scalars.append(0.0)
        else:
            he_grad_scalars.append(decrypt_scalar(encrypted_coxcc_grads[s]))

    plain_weight_grads_accum = None
    for s in range(n_samples):
        if encrypted_coxcc_grads[s] is None:
            continue
        _, intermediates = plaintext_forward(model_plain, X[s:s+1])
        grad_np = np.ones((1, dims[-1])) * he_grad_scalars[s]
        sample_g = plaintext_backward(model_plain, intermediates, grad_np)
        if plain_weight_grads_accum is None:
            plain_weight_grads_accum = [g.copy() for g in sample_g]
        else:
            for i in range(n_layers):
                plain_weight_grads_accum[i] += sample_g[i]

    if plain_weight_grads_accum is not None:
        plain_weights = get_weights(model_plain)
        for i in range(n_layers):
            update = (lr / n_samples) * plain_weight_grads_accum[i]
            plain_weights[i] -= update
        set_weights(model_plain, plain_weights)

    if validate_all or stage == 'update':
        print(f"\n  Verifying forward after weight update...")
        for s in range(n_samples):
            z_he = client_models[0].forward(X[s:s+1])
            score_he = decrypt_scalar(z_he[-1])
            score_plain, _ = plaintext_forward(model_plain, X[s:s+1])
            score_plain = score_plain.item() if hasattr(score_plain, 'item') \
                else float(score_plain[0])
            ok = check(f"Post-update score[{s}]", score_he, score_plain)
            if not ok:
                all_ok = False

        if stage == 'update':
            return print_result(all_ok)

    if target_idx < _stage_idx('multi_iter'):
        return print_result(all_ok)

    print(f"\n--- Phase 5: Multi-iter ({args.n_iters} iterations) ---")

    for c in range(n_clients):
        set_weights(client_models[c], [w.copy() for w in initial_weights])
    set_weights(model_plain, [w.copy() for w in initial_weights])

    multi_ok = run_multi_iter(client_models[0], model_plain, X, events, dims,
                               n_layers, n_samples, n_clients, args.n_iters,
                               lr, do_bootstrap, pathway_mask, margs,
                               survival_times=survival_times,
                               sigmoid_poly_degree=args.sigmoid_poly_degree,
                               seed=args.seed)
    if not multi_ok:
        all_ok = False

    return print_result(all_ok)

def run_multi_iter(model, model_plain, X, events, dims, n_layers,
                    n_samples, n_clients, n_iters, lr, do_bootstrap,
                    pathway_mask, margs, survival_times=None,
                    sigmoid_poly_degree=SIGMOID_POLY_DEGREE, seed=42):
    from he.trainer import train_epoch_he

    piece_size = n_samples // n_clients
    client_data = None
    if n_clients > 1:
        client_data = []
        for c in range(n_clients):
            start = c * piece_size
            end = (c + 1) * piece_size if c < n_clients - 1 else n_samples
            client_data.append((
                X[start:end],
                survival_times[start:end],
                events[start:end],
            ))

    effective_lr = lr / n_samples
    all_ok = True
    momentum = model_plain.momentum
    plain_x_prev = None

    for iteration in range(n_iters):
        is_last = (iteration == n_iters - 1)
        print(f"\n  {'#'*55}")
        print(f"  # ITERATION {iteration + 1}/{n_iters}")
        print(f"  {'#'*55}")

        iter_t0 = time.time()
        coxcc_seed = seed + 1000 + iteration

        np.random.seed(coxcc_seed)
        train_epoch_he(
            model, X, survival_times, events, lr,
            n_clients=n_clients, client_data=client_data,
            bootstrap=do_bootstrap,
            sigmoid_interval=SIGMOID_INTERVAL, sigmoid_degree=sigmoid_poly_degree,
            is_last_iteration=(is_last and not do_bootstrap),
        )
        print(f"    train_epoch_he: done")

        plain_x = get_weights(model_plain)
        if momentum > 0.0 and plain_x_prev is not None:
            plain_y = [plain_x[i] + momentum * (plain_x[i] - plain_x_prev[i])
                       for i in range(n_layers)]
            set_weights(model_plain, plain_y)

        plain_scores = []
        for s in range(n_samples):
            scores, _ = plaintext_forward(model_plain, X[s:s+1])
            plain_scores.append(scores.item() if hasattr(scores, 'item')
                                else float(scores[0]))
        plain_scores_arr = np.array(plain_scores)

        np.random.seed(coxcc_seed)
        plain_loss_grads = np.zeros(n_samples)
        if n_clients > 1 and client_data is not None:
            for c, (X_c, dur_c, evt_c) in enumerate(client_data):
                c_start = sum(len(client_data[k][0]) for k in range(c))
                pairs = pair_events_controls(dur_c, evt_c)
                for i_evt, j_ctrl in pairs:
                    diff = plain_scores_arr[c_start + j_ctrl] - plain_scores_arr[c_start + i_evt]
                    sig = sigmoid_plain(diff, SIGMOID_INTERVAL, sigmoid_poly_degree)
                    plain_loss_grads[c_start + j_ctrl] += sig
                    plain_loss_grads[c_start + i_evt] -= sig
        else:
            pairs = pair_events_controls(survival_times, events)
            for i_evt, j_ctrl in pairs:
                diff = plain_scores_arr[j_ctrl] - plain_scores_arr[i_evt]
                sig = sigmoid_plain(diff, SIGMOID_INTERVAL, sigmoid_poly_degree)
                plain_loss_grads[j_ctrl] += sig
                plain_loss_grads[i_evt] -= sig

        plain_grads_accum = None
        for s in range(n_samples):
            if plain_loss_grads[s] == 0.0:
                continue
            _, intermediates = plaintext_forward(model_plain, X[s:s+1])
            grad_np = np.ones((1, dims[-1])) * plain_loss_grads[s]
            sample_grads = plaintext_backward(model_plain, intermediates, grad_np)
            if plain_grads_accum is None:
                plain_grads_accum = [g.copy() for g in sample_grads]
            else:
                for i in range(n_layers):
                    plain_grads_accum[i] += sample_grads[i]

        if plain_grads_accum is not None:
            plain_y_now = get_weights(model_plain)
            plain_x_new = []
            for i in range(n_layers):
                update = effective_lr * plain_grads_accum[i]
                plain_x_new.append(plain_y_now[i] - update)
            if momentum > 0.0:
                plain_x_prev = plain_x
            set_weights(model_plain, plain_x_new)

        print(f"    Time: {time.time()-iter_t0:.1f}s")

    print(f"\n  Final validation after {n_iters} iterations:")

    he_final_scores = []
    for s in range(n_samples):
        z_he = model.forward(X[s:s+1])
        he_final_scores.append(decrypt_scalar(z_he[-1]))

    plain_final_scores = []
    for s in range(n_samples):
        scores, _ = plaintext_forward(model_plain, X[s:s+1])
        plain_final_scores.append(scores.item() if hasattr(scores, 'item')
                                  else float(scores[0]))

    max_diff = 0
    for s in range(n_samples):
        diff = abs(he_final_scores[s] - plain_final_scores[s])
        max_diff = max(max_diff, diff)
        ok = check(f"final_score[{s}]", he_final_scores[s], plain_final_scores[s],
                   threshold=1e-3)
        if not ok:
            all_ok = False

    score_ok = max_diff < 1e-3
    if not score_ok:
        all_ok = False
    print(f"  Max score diff: {max_diff:.2e} [{'OK' if score_ok else 'FAIL'}]")

    return all_ok

def main():
    parser = argparse.ArgumentParser(
        description="Encrypted pipeline test (Cox-CC loss, 37->138->1)")
    parser.add_argument("--stage", type=str, required=True,
                        choices=STAGES + ['all'],
                        help="Pipeline stage to validate")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--n-iters", type=int, default=3,
                        help="Number of iterations (multi_iter only)")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ring-dim-exp", type=int, default=None,
                        help="Ring dimension exponent. Unset: 128-bit default (2^17 K=1, 2^16 K>1). "
                             "Pass 15 for K=1 experiments to fit in RAM (not 128-bit).")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Enable FHE bootstrapping (required for multi_iter/all)")
    parser.add_argument("--dims", type=int, nargs='+',
                        default=DEFAULT_DIMS,
                        help=f"Network dimensions (default: {' '.join(map(str, DEFAULT_DIMS))})")
    parser.add_argument("--approx-degree", type=int, default=APPROX_DEGREE,
                        help=f"Activation polynomial degree (default: {APPROX_DEGREE})")
    parser.add_argument("--n-clients", type=int, default=1,
                        help="Number of federated clients (default: 1)")
    parser.add_argument("--sigmoid-poly-degree", type=int, default=APPROX_DEGREE,
                        help=f"sigmoid_he polynomial degree (default: {APPROX_DEGREE})")
    parser.add_argument("--verbose", type=int, default=0,
                        help="Verbosity level (1+ enables level logging)")
    args = parser.parse_args()

    if args.stage in ('multi_iter', 'all') and not args.bootstrap:
        parser.error("--stage multi_iter/all requires --bootstrap (the no-bootstrap "
                     "context cannot reliably decode multi-iteration chains at ring 2^15)")

    if args.n_clients > 1 and not args.bootstrap:
        parser.error("--n-clients > 1 requires --bootstrap (multi-party needs FHE)")

    if args.n_clients > args.n_samples:
        parser.error("--n-clients cannot exceed --n-samples")

    success = run_pipeline(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
