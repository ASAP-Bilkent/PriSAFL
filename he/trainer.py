from he.prisafltext import PriSAFLText
from he.coxcc_loss import compute_coxcc_gradient, pair_events_controls
from he.utils import timer_context

def train_epoch_he(model, X_train, durations, events, learning_rate,
                   n_clients=1, client_data=None, client_pairs=None,
                   bootstrap=False,
                   sigmoid_interval=15.0, sigmoid_degree=3,
                   is_last_iteration=False):
    n_samples = X_train.shape[0]
    model.apply_lookahead()

    if n_clients > 1 and client_data is not None:
        all_gradients = []
        for c, (X_c, dur_c, evt_c) in enumerate(client_data):
            grads = _train_single_client(
                model, X_c, dur_c, evt_c,
                pairs=client_pairs[c] if client_pairs is not None else None,
                sigmoid_interval=sigmoid_interval, sigmoid_degree=sigmoid_degree,
            )
            all_gradients.append(grads)

        with timer_context("Aggregation"):
            avg_gradients = []
            for layer_idx in range(len(model.layers)):
                layer_grads = [g[layer_idx] for g in all_gradients if g[layer_idx] is not None]
                if layer_grads:
                    avg = PriSAFLText.sum_ciphertexts(layer_grads)
                    avg_gradients.append(avg)
                else:
                    avg_gradients.append(all_gradients[0][layer_idx])
    else:
        avg_gradients = _train_single_client(
            model, X_train, durations, events,
            sigmoid_interval=sigmoid_interval, sigmoid_degree=sigmoid_degree,
        )

    effective_lr = learning_rate / n_samples
    model.update_weights(avg_gradients, effective_lr,
                         is_last_iteration=is_last_iteration,
                         bootstrap=bootstrap)

def _log_level(label, ht):
    if ht is not None and PriSAFLText.VERBOSE >= 1:
        lvl = ht.get_level()
        if lvl is not None:
            print(f"    [level] {label}: {lvl}")

def _train_single_client(model, X, durations, events, pairs=None,
                         sigmoid_interval=15.0, sigmoid_degree=3):
    n = X.shape[0]

    if pairs is None:
        pairs = pair_events_controls(durations, events)
    if not pairs:
        return [None] * len(model.layers)
    participants = sorted({idx for pair in pairs for idx in pair})

    with timer_context("Forward"):
        all_z = [None] * n
        all_derivs = [None] * n
        for s in participants:
            z = model.forward(X[s:s+1])
            all_z[s] = z
            all_derivs[s] = [layer.preZ_deriv for layer in model.layers]
        for i, zi in enumerate(all_z[participants[0]]):
            _log_level(f"forward L{i+1} output", zi)

    risk_scores_he = [z[-1] if z is not None else None for z in all_z]

    with timer_context("CoxCCGradient"):
        grad_he_list = compute_coxcc_gradient(
            risk_scores_he, pairs,
            sigmoid_interval=sigmoid_interval, sigmoid_degree=sigmoid_degree,
        )

    first_grad = next((g for g in grad_he_list if g is not None), None)
    _log_level("coxcc gradient", first_grad)

    with timer_context("Backward"):
        all_layer_grads = []
        for s in participants:
            if grad_he_list[s] is None:
                continue
            for layer, deriv in zip(model.layers, all_derivs[s]):
                layer.preZ_deriv = deriv
            g = list(all_z[s])
            g[-1] = grad_he_list[s]
            grads = model.backward(X[s:s+1], g)
            if s == participants[0]:
                for i, dw in enumerate(grads):
                    _log_level(f"backward L{i+1} weight_grad", dw)
            all_layer_grads.append(grads)

    if not all_layer_grads:
        return [None] * len(model.layers)

    avg_grads = []
    for layer_idx in range(len(model.layers)):
        layer_grads = [g[layer_idx] for g in all_layer_grads]
        avg_grads.append(PriSAFLText.sum_ciphertexts(layer_grads))

    return avg_grads
