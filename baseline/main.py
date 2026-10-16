import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from model import Net, initialize_weights

from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss
from pycox.evaluation import EvalSurv
import torchtuples as tt
import torch
from sklearn.model_selection import KFold
import numpy as np
from load_targeted_data import predicted_quant, events, survival, pathway_info
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from model_utils import EarlyStopping

SEEDS = [6, 35, 81]
learningrate = 0.0001
epochnum = 5000
patience = 150
l2_lambda = 0.002

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOGS_DIR = SCRIPT_DIR.parent / "logs" / "baseline"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--nondeterministic", action="store_true",
                help="Disable deterministic seeding (seeds are fixed by default)")
_p.add_argument("--output", default=None,
                help="Output CSV name (without .csv) under ablation/logs/ (default: writes to baseline logs)")
_p.add_argument("--patient-grouped", action="store_true",
                help="Split outer folds and the inner validation split by patient group "
                     "(default: by spectrum, as in PiDeeL)")
_p.add_argument("--verified-only", action="store_true",
                help="With --patient-grouped: keep only spectra whose patient ID is checksum-verified "
                     "against the earlier Zenodo releases (column verified, 339 spectra)")
_args = _p.parse_args()

if _args.patient_grouped:
    import pandas as pd
    _pg = pd.read_csv(SCRIPT_DIR.parent / "data" / "patient_group.csv")
    patient_ids = _pg["group"].values
    keep = _pg["verified"].values == 1 if _args.verified_only else np.ones(len(patient_ids), bool)
    groups = np.unique(patient_ids[keep])

c_indices = []
for SEED in SEEDS:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    if _args.patient_grouped:
        result = [(np.flatnonzero(np.isin(patient_ids, groups[g_tr]) & keep),
                   np.flatnonzero(np.isin(patient_ids, groups[g_te]) & keep))
                  for g_tr, g_te in kf.split(groups)]
    else:
        result = list(kf.split(events))
    for i in range(len(result)):
        if not _args.nondeterministic:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)

        X_train = predicted_quant[result[i][0]]
        X_test = predicted_quant[result[i][1]]
        labels = np.hstack((events, survival))
        y_train = labels[result[i][0]]
        y_test = labels[result[i][1]]
        
        if _args.patient_grouped:
            pid_train = patient_ids[result[i][0]]
            g_fit, _ = train_test_split(np.unique(pid_train), test_size=0.2, random_state=SEED)
            fit = np.isin(pid_train, g_fit)
            assert not set(pid_train) & set(patient_ids[result[i][1]])
            X_train, X_cv, y_train, y_cv = X_train[fit], X_train[~fit], y_train[fit], y_train[~fit]
        else:
            X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_cv = scaler.transform(X_cv)
        X_test = scaler.transform(X_test)
        pathway_info_t = torch.as_tensor(pathway_info).to(device)

        model = Net(X_train.shape[1], 1, pathway_info_t).to(device)
        model.apply(initialize_weights)
        optim = torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=1e-5)
        early_stopper = EarlyStopping(patience=patience, path=str(LOGS_DIR / 'checkpoint_survival.pt'))
        criterion = CoxPHLoss()
        
        train_losses = []
        vald_losses = []
        for epoch in range(1, epochnum + 1):
            all_features = torch.tensor(X_train).float().to(device)
            all_labels = torch.tensor(y_train).float().to(device)

            optim.zero_grad()
            pred_probas = model(all_features)
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = criterion(pred_probas, all_labels[:, 1], all_labels[:, 0]) + l2_norm * l2_lambda
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

            with torch.no_grad():
                features_cv = torch.tensor(X_cv).float().to(device)
                labels_cv = torch.tensor(y_cv).float().to(device)
                pred_probas = model(features_cv)
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                vloss = criterion(pred_probas, labels_cv[:, 1], labels_cv[:, 0]) + l2_norm * l2_lambda
            vald_losses.append(vloss.item())

            print(f"Epoch {epoch}, Train loss: {train_losses[-1]:.6f}, Vald loss: {vald_losses[-1]:.6f}")

            early_stopper(vald_losses[-1], model)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(torch.load(str(LOGS_DIR / 'checkpoint_survival.pt')))
        features_test = torch.tensor(X_test).float().to(device)

        model1 = CoxPH(model, tt.optim.Adam)
        _ = model1.compute_baseline_hazards(input=all_features, target=(all_labels[:, 1], all_labels[:, 0]))
        surv = model1.predict_surv_df(features_test)
        ev = EvalSurv(surv, y_test[:, 1], y_test[:, 0], censor_surv='km')
        c_indices.append(ev.concordance_td())

if _args.output:
    _out_dir = SCRIPT_DIR.parent / "ablation" / "logs"
    _out_dir.mkdir(parents=True, exist_ok=True)
    _out_path = _out_dir / f"{_args.output}.csv"
else:
    _out_path = LOGS_DIR / "c_indices.csv"

with open(_out_path, "w") as f:
    f.write("seed,fold,c_index\n")
    idx = 0
    for SEED in SEEDS:
        for fold in range(5):
            f.write(f"{SEED},{fold},{c_indices[idx]:.6f}\n")
            idx += 1

print(f"\nMean C-index: {np.mean(c_indices):.4f} +/- {np.std(c_indices):.4f}")
print(f"Median C-index: {np.median(c_indices):.4f}")
print(f"Results saved to {_out_path}")
