import numpy as np
from he.sigmoid_he import sigmoid_he

def pair_events_controls(durations, events):
    n = len(durations)
    event_indices = np.where(events.flatten() == 1)[0]
    pairs = []
    for i in event_indices:
        candidates = np.where(
            (durations.flatten() >= durations.flatten()[i]) &
            (np.arange(n) != i)
        )[0]
        if len(candidates) > 0:
            j = np.random.choice(candidates)
            pairs.append((int(i), int(j)))
    return pairs

def compute_coxcc_gradient(risk_scores_he, pairs,
                            sigmoid_interval=15.0, sigmoid_degree=3):
    grad_accumulators = [None] * len(risk_scores_he)

    for i_evt, j_ctrl in pairs:
        diff = risk_scores_he[j_ctrl].subtract(risk_scores_he[i_evt], "CC")
        sig = sigmoid_he(diff, interval=sigmoid_interval, poly_degree=sigmoid_degree)

        neg_sig = sig.negate()

        if grad_accumulators[j_ctrl] is None:
            grad_accumulators[j_ctrl] = sig
        else:
            grad_accumulators[j_ctrl] = grad_accumulators[j_ctrl].add(sig, "CC")

        if grad_accumulators[i_evt] is None:
            grad_accumulators[i_evt] = neg_sig
        else:
            grad_accumulators[i_evt] = grad_accumulators[i_evt].add(neg_sig, "CC")

    return grad_accumulators
