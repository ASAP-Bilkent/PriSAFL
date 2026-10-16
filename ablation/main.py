import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn import preprocessing

from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss
from pycox.evaluation import EvalSurv
import torchtuples as tt

from load_data import predicted_quant, events, survival, pathway_info as pathway_info_full
from model import Net, make_init_fn
from approx import _make_approx_sigmoid, _make_approx_coxph
from losses import ApproxCoxPHLoss, CoxCCLoss, CoxCCPreSampledLoss, CoxCCMiniBatchLoss
from sampling import coxcc_sample_pairs, coxcc_sample_batch, partition_by_events, partition_dirichlet

SEEDS = [6, 35, 81]

def parse_args():
    p = argparse.ArgumentParser(description="Ablation training (survival, SGD, no bias)")
    p.add_argument("--activation", default="3-15",
                   help="'relu' or polynomial spec like '3-15' (degree-3, interval 15)")
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Mini-batch size (None=full batch)")
    p.add_argument("--loss", choices=["coxph", "cox-cc"], default="coxph")
    p.add_argument("--n-events", type=int, default=1,
                   help="Number of event patients per Cox-CC mini-batch")
    p.add_argument("--n-control", type=int, default=1,
                   help="Cox-CC controls sampled per event each step (loss averaged over them)")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 0.15 for cox-cc, 0.035 for coxph)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--n-clients", type=int, default=1,
                   help="Number of federated clients (1=single network, >1=federated simulation)")
    p.add_argument("--approx-cox-cc", default=None,
                   help="Cox-CC gradient sigmoid: '<deg>-<interval>[c][T<tau>]' or 'exact', e.g. '3-15cT4'")
    p.add_argument("--approx-coxph", default=None,
                   help="CoxPH exp/inv: '<deg>-<k>' or HE spec, e.g. '7-2-m-l8-u8-b16-r2.5-n4-s7'. '' for exact")
    p.add_argument("--eval-at", default=None,
                   help="Comma-separated epochs to evaluate at. run_survival then returns {epoch: c_indices}")
    p.add_argument("--patient-grouped", action="store_true",
                   help="Split outer folds, the inner held-out 20%% and the federated clients by "
                        "patient group (data/patient_group.csv). Default: by spectrum, as in PiDeeL")
    p.add_argument("--verified-only", action="store_true",
                   help="With --patient-grouped: keep only spectra whose patient ID is checksum-verified "
                        "against the earlier Zenodo releases (column verified, 339 spectra)")
    p.add_argument("--dirichlet-beta", type=float, default=None,
                   help="Non-IID clients: Dirichlet label skew over event x survival-median strata "
                        "(default: event-stratified IID split)")
    p.add_argument("--local-norm", action="store_true",
                   help="Each client z-scores its training data with its own mean/std "
                        "(default: pooled = federated statistics). Test data always use pooled statistics")
    p.add_argument("--nondeterministic", action="store_true",
                   help="Disable deterministic seeding (seeds are fixed by default)")
    p.add_argument("--output", default=None,
                   help="Output CSV name (without .csv) under ablation/logs/ (default: c_indices)")
    args = p.parse_args()
    if args.lr is None:
        args.lr = 0.15 if args.loss == "cox-cc" else 0.035
    if args.loss == "cox-cc" and args.approx_cox_cc is None:
        args.approx_cox_cc = args.activation
    if args.loss == "coxph" and args.approx_coxph is None and args.activation != "relu":
        args.approx_coxph = f"{args.activation.split('-')[0]}-5"
    return args

def _inner_keep(groups, seed):
    keep_g, _ = train_test_split(np.unique(groups), test_size=0.2, random_state=seed)
    return np.isin(groups, keep_g)

def _load_fold(train_idx, test_idx, seed, groups=None):
    X_train = predicted_quant[train_idx]
    X_test = predicted_quant[test_idx]
    labels = np.hstack((events, survival))
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    if groups is None:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed)
    else:
        keep = _inner_keep(groups, seed)
        X_train, y_train = X_train[keep], y_train[keep]
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def _log_partition(client_data):
    sizes = [len(cy) for _, cy in client_data]
    no_ev = sum(cy[:, 0].sum() == 0 for _, cy in client_data)
    n_ev = sum(int(cy[:, 0].sum()) for _, cy in client_data)
    no_ctrl = sum(int(((cy[:, 1] >= cy[i, 1]).sum() - 1) == 0)
                  for _, cy in client_data for i in np.flatnonzero(cy[:, 0] == 1))
    print(f"PARTITION min {min(sizes)} max {max(sizes)} no_event_clients {no_ev} "
          f"events {n_ev} events_without_control {no_ctrl}", flush=True)

def _seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _build_model(args, input_dim, pw_t, act_func, device):
    model = Net(input_dim, 1, pw_t, act_func=act_func, n_layers=args.n_layers).to(device)
    model.apply(make_init_fn())
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2,
                            momentum=args.momentum, nesterov=args.momentum > 0)
    return model, optim

def _build_criterion(args):
    approx_sig_fn = _make_approx_sigmoid(args.approx_cox_cc) if args.approx_cox_cc else None
    if args.loss == "cox-cc":
        return CoxCCLoss(approx_sigmoid_fn=approx_sig_fn, n_control=args.n_control), approx_sig_fn
    if args.approx_coxph:
        approx_exp_fn, approx_inv_fn = _make_approx_coxph(args.approx_coxph)
        return ApproxCoxPHLoss(approx_exp_fn, approx_inv_fn), approx_sig_fn
    return CoxPHLoss(), approx_sig_fn

def _scale_grads(model, denom):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.div_(denom)

def _make_step(args, model, criterion, approx_sig_fn, all_features, all_labels,
               X_train, y_train, client_data, client_features, client_labels, device):
    if args.n_clients > 1 and args.loss == "cox-cc":
        presampled = [CoxCCPreSampledLoss(1, approx_sigmoid_fn=approx_sig_fn) for cd in client_data]
        def step():
            for cd, ps_criterion in zip(client_data, presampled):
                ev_local, ctrl_local, unique_idx = coxcc_sample_pairs(cd[1][:, 1], cd[1][:, 0], args.n_control)
                if ev_local is None:
                    continue
                subset_features = torch.tensor(cd[0][unique_idx]).float().to(device)
                ev_t = torch.tensor(ev_local, device=device)
                ctrl_t = torch.tensor(ctrl_local, device=device)
                pred = model(subset_features)
                ps_criterion(pred, ev_t, ctrl_t).backward()
            _scale_grads(model, len(X_train) * args.n_control)
            return True
        return step

    if args.n_clients > 1:
        n_events_total = sum(cl[:, 0].sum() for cl in client_labels)
        def step():
            for cf, cl in zip(client_features, client_labels):
                if cl[:, 0].sum() == 0:
                    continue
                pred = model(cf)
                loss = criterion(pred, cl[:, 1], cl[:, 0])
                loss = loss * cl[:, 0].sum()
                loss.backward()
            _scale_grads(model, n_events_total)
            return True
        return step

    if args.loss == "cox-cc" and args.batch_size is not None:
        mb_criterion = CoxCCMiniBatchLoss(n_events=args.n_events)
        def step():
            n_controls = args.batch_size - args.n_events
            batch_idx = coxcc_sample_batch(
                y_train[:, 1], y_train[:, 0], n_controls, n_events=args.n_events)
            if batch_idx is None:
                return False
            mb_criterion(model(all_features[batch_idx])).backward()
            return True
        return step

    if args.batch_size is not None and args.batch_size < X_train.shape[0]:
        def step():
            idx = np.sort(np.random.permutation(X_train.shape[0])[:args.batch_size])
            batch_labels = all_labels[idx]
            criterion(model(all_features[idx]), batch_labels[:, 1], batch_labels[:, 0]).backward()
            return True
        return step

    def step():
        pred = model(all_features)
        criterion(pred, all_labels[:, 1], all_labels[:, 0]).backward()
        return True
    return step

def _evaluate(model, all_features, all_labels, X_test, y_test, device):
    model1 = CoxPH(model, tt.optim.Adam)
    model1.compute_baseline_hazards(
        input=all_features, target=(all_labels[:, 1], all_labels[:, 0]))
    features_test = torch.tensor(X_test).float().to(device)
    surv = model1.predict_surv_df(features_test)
    ev = EvalSurv(surv, y_test[:, 1], y_test[:, 0], censor_surv='km')
    return ev.concordance_td()

def run_survival(args, pathway_info, device):
    pw_t = torch.as_tensor(pathway_info).to(device)
    act_func = None if args.activation == "relu" else args.activation
    c_indices = []
    eval_at = sorted(int(e) for e in args.eval_at.split(",")) if args.eval_at else [args.epochs]
    by_epoch = {e: [] for e in eval_at}
    if args.patient_grouped:
        import pandas as pd
        pg = pd.read_csv(SCRIPT_DIR.parent / "data" / "patient_group.csv")
        pid = pg["group"].values
        keep = pg["verified"].values == 1 if args.verified_only else np.ones(len(pid), bool)
        ug = np.unique(pid[keep])
    for SEED in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        if args.patient_grouped:
            splits = [(np.flatnonzero(np.isin(pid, ug[g_tr]) & keep), np.flatnonzero(np.isin(pid, ug[g_te]) & keep))
                      for g_tr, g_te in kf.split(ug)]
        else:
            splits = kf.split(events)
        for train_idx, test_idx in splits:
            g_fold = g_train = None
            if args.patient_grouped:
                g_fold = pid[train_idx]
                assert not set(g_fold) & set(pid[test_idx])
                g_train = g_fold[_inner_keep(g_fold, SEED)]
            X_train, X_test, y_train, y_test = _load_fold(train_idx, test_idx, SEED, groups=g_fold)

            if not args.nondeterministic:
                _seed_everything(SEED)

            model, optim = _build_model(args, X_train.shape[1], pw_t, act_func, device)

            client_data = client_features = client_labels = None
            if args.n_clients > 1:
                if args.dirichlet_beta is not None:
                    assert g_train is None, "Dirichlet split is spectrum-level only"
                    client_data = partition_dirichlet(X_train, y_train, args.n_clients, args.dirichlet_beta)
                else:
                    client_data = partition_by_events(X_train, y_train, args.n_clients, groups=g_train)
                if args.local_norm:
                    client_data = [(preprocessing.StandardScaler().fit_transform(cx), cy) for cx, cy in client_data]
                _log_partition(client_data)
                client_features = [torch.tensor(cd[0]).float().to(device) for cd in client_data]
                client_labels = [torch.tensor(cd[1]).float().to(device) for cd in client_data]

            all_features = torch.tensor(X_train).float().to(device)
            all_labels = torch.tensor(y_train).float().to(device)

            criterion, approx_sig_fn = _build_criterion(args)
            step = _make_step(args, model, criterion, approx_sig_fn, all_features, all_labels,
                              X_train, y_train, client_data, client_features, client_labels, device)

            for epoch in range(1, max(eval_at) + 1):
                optim.zero_grad()
                if step():
                    optim.step()
                if epoch in by_epoch:
                    by_epoch[epoch].append(_evaluate(model, all_features, all_labels, X_test, y_test, device))

    return by_epoch if args.eval_at else by_epoch[args.epochs]

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    LOGS_DIR = SCRIPT_DIR / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    act_label = args.activation
    hdim_label = pathway_info_full.shape[1]
    bs_label = args.batch_size if args.batch_size is not None else "full"

    print(f"Act: {act_label} | Hdim: {hdim_label} | Batch: {bs_label} | "
          f"Loss: {args.loss} | Mom: {args.momentum} | Epochs: {args.epochs} | "
          f"Clients: {args.n_clients}")

    c_indices = run_survival(args, pathway_info_full, device)

    csv_name = f"{args.output}.csv" if args.output else "c_indices.csv"
    with open(LOGS_DIR / csv_name, "w") as f:
        f.write("seed,fold,act_func,n_layers,hidden_dim,batch_size,loss_type,n_clients,c_index\n")
        idx = 0
        for SEED in SEEDS:
            for fold in range(5):
                f.write(f"{SEED},{fold},{act_label},{args.n_layers},"
                        f"{hdim_label},{bs_label},{args.loss},{args.n_clients},{c_indices[idx]:.6f}\n")
                idx += 1

    print(f"\nMean C-index: {np.mean(c_indices):.4f} +/- {np.std(c_indices):.4f}")
    print(f"Median C-index: {np.median(c_indices):.4f}")
    print(f"Results saved to {LOGS_DIR}")

if __name__ == "__main__":
    main()
