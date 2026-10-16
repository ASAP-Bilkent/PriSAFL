import numpy as np

def coxcc_sample_pairs(durations_np, events_np, n_control=1):
    n = len(durations_np)
    event_indices = np.where(events_np.flatten() == 1)[0]
    event_list = []
    control_list = []
    for i in event_indices:
        candidates = np.where((durations_np.flatten() >= durations_np.flatten()[i])
                              & (np.arange(n) != i))[0]
        if len(candidates) == 0:
            continue
        ctrl = [np.random.choice(candidates)] if n_control == 1 else np.random.choice(candidates, n_control)
        event_list += [i] * len(ctrl)
        control_list += list(ctrl)

    if len(event_list) == 0:
        return None, None, None

    unique_idx = np.unique(np.concatenate([event_list, control_list]))
    remap = {orig: new for new, orig in enumerate(unique_idx)}
    event_local = np.array([remap[i] for i in event_list])
    control_local = np.array([remap[j] for j in control_list])
    return event_local, control_local, unique_idx

def coxcc_sample_batch(durations, events, n_controls=9, n_events=1):
    dur_np = durations if isinstance(durations, np.ndarray) else durations.numpy()
    evt_np = events if isinstance(events, np.ndarray) else events.numpy()

    event_indices = np.where(evt_np.flatten() == 1)[0]
    if len(event_indices) == 0:
        return None

    n_events = min(n_events, len(event_indices))
    chosen_events = np.random.choice(event_indices, n_events, replace=False)

    all_candidates = set()
    for event_idx in chosen_events:
        candidates = np.where((dur_np.flatten() >= dur_np.flatten()[event_idx])
                              & ~np.isin(np.arange(len(dur_np)), chosen_events))[0]
        all_candidates.update(candidates)

    if len(all_candidates) == 0:
        return None

    all_candidates = np.array(list(all_candidates))
    if len(all_candidates) >= n_controls:
        control_indices = np.random.choice(all_candidates, n_controls, replace=False)
    else:
        control_indices = np.random.choice(all_candidates, n_controls, replace=True)

    return np.concatenate([chosen_events, control_indices])

def partition_by_events(X, y, n_clients, groups=None):
    events_col = y[:, 0]
    by_spectrum = groups is None
    if by_spectrum:
        groups = np.arange(len(y))
    ug = np.unique(groups)
    ev_g = np.array([events_col[groups == g].max() == 1 for g in ug])
    event_idx = ug[ev_g]
    nonevent_idx = ug[~ev_g]
    np.random.shuffle(event_idx)
    np.random.shuffle(nonevent_idx)

    event_splits = np.array_split(event_idx, n_clients)
    nonevent_splits = np.array_split(nonevent_idx, n_clients)

    client_data = []
    for c in range(n_clients):
        idx = np.concatenate([event_splits[c], nonevent_splits[c]])
        if not by_spectrum:
            idx = np.flatnonzero(np.isin(groups, idx))
        np.random.shuffle(idx)
        client_data.append((X[idx], y[idx]))
    return client_data

def partition_dirichlet(X, y, n_clients, beta):
    strata = 2 * y[:, 0].astype(int) + (y[:, 1] >= np.median(y[:, 1])).astype(int)
    while True:
        parts = [[] for _ in range(n_clients)]
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            np.random.shuffle(idx)
            cuts = (np.cumsum(np.random.dirichlet([beta] * n_clients)) * len(idx)).astype(int)[:-1]
            for c, chunk in enumerate(np.split(idx, cuts)):
                parts[c] += chunk.tolist()
        if min(len(p) for p in parts) > 0:
            break
    client_data = []
    for p in parts:
        idx = np.array(p)
        np.random.shuffle(idx)
        client_data.append((X[idx], y[idx]))
    return client_data
