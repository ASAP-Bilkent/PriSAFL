import argparse
import pickle
import numpy as np
from pathlib import Path
from types import SimpleNamespace

PRISAFL_ROOT = Path(__file__).parent.parent
DATA_DIR = PRISAFL_ROOT / "data"

DIMS = [37, 138, 1]
ACT_FUNC = 'approx_relu'
APPROX_DEGREE = 3
APPROX_INTERVAL = 15
LR = 0.02
EPOCHS = 100
BATCH_SIZE = 1
VERBOSITY = 0
MOMENTUM = 0.9

def load_data():
    with open(DATA_DIR / 'predicted_quant.pickle', 'rb') as f:
        predicted_quant = pickle.load(f)
    with open(DATA_DIR / 'censorship.pickle', 'rb') as f:
        censorship = pickle.load(f)
    with open(DATA_DIR / 'survival.pickle', 'rb') as f:
        survival = pickle.load(f)

    events = np.array([1 if c == 1 else 0 for c in censorship], dtype=np.float64)
    survival = np.array(survival, dtype=np.float64).flatten()

    pathway_info = []
    with open(DATA_DIR / "result.csv") as f:
        for index, line in enumerate(f):
            if index == 0:
                continue
            tokens = line.split(",")
            pathway_info.append(tokens[2:])
    pathway_info = np.array(pathway_info, dtype=np.int64).reshape(37, -1)

    return predicted_quant, events, survival, pathway_info

def parse_args():
    p = argparse.ArgumentParser(description="Pri-SAFL encrypted Cox-CC training")
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--momentum", type=float, default=MOMENTUM)
    p.add_argument("--n-clients", type=int, default=1)
    p.add_argument("--ring-dim-exp", type=int, default=None,
                   help="Unset: 128-bit default (2^17 K=1, 2^16 K>1). 15 fits K=1 in RAM (not 128-bit).")
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--activation", default=f"{APPROX_DEGREE}-{APPROX_INTERVAL}",
                   help="Polynomial activation spec (e.g., '3-15' = degree 3, interval 15)")
    p.add_argument("--dims", type=int, nargs="+", default=DIMS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbosity", type=int, default=VERBOSITY)
    p.add_argument("--timing", action="store_true",
                   help="Print per-phase timing table at the end")
    return p.parse_args()

def main():
    args = parse_args()

    from sklearn.preprocessing import StandardScaler

    print("Loading data...")
    predicted_quant, events, survival, pathway_info = load_data()

    np.random.seed(args.seed)

    scaler = StandardScaler().fit(predicted_quant[:args.n_samples])
    X = scaler.transform(predicted_quant[:args.n_samples])
    dur = survival[:args.n_samples]
    evt = events[:args.n_samples]

    print(f"Samples: {args.n_samples}, Events: {int(evt.sum())}, "
          f"Epochs: {args.epochs}, LR: {args.lr}, Clients: {args.n_clients}")

    act_parts = args.activation.split("-")
    model_args = SimpleNamespace(
        act_func="approx_relu",
        approx_degree=int(act_parts[0]),
        approx_b=float(act_parts[1]),
        batch_size=1,
        lr=args.lr,
        momentum=args.momentum,
        verbosity=args.verbosity,
    )

    print("Setting up CKKS context...")
    from he.encryption import CKKSContext
    from he.prisafltext import PriSAFLText
    from he.model import PriSAFLHE
    from he.trainer import train_epoch_he
    from he.utils import global_timer

    ckks_ctx = CKKSContext(n_clients=args.n_clients, ring_dim_exp=args.ring_dim_exp)
    PriSAFLText.set_ckks_context(ckks_ctx)
    print(f"Ring dim: {ckks_ctx.cc.GetRingDimension()}, Depth: {ckks_ctx.depth}")

    model = PriSAFLHE(args.dims, pathway_info, model_args)

    client_data = None
    if args.n_clients > 1:
        piece = args.n_samples // args.n_clients
        client_data = []
        for c in range(args.n_clients):
            start = c * piece
            end = start + piece if c < args.n_clients - 1 else args.n_samples
            client_data.append((X[start:end], dur[start:end], evt[start:end]))

    if args.timing:
        global_timer.set_n_clients(args.n_clients)
        global_timer.set_batch_info(args.n_samples, 1)
        global_timer.set_total_iterations(args.epochs)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_epoch_he(
            model, X, dur, evt, args.lr,
            n_clients=args.n_clients,
            client_data=client_data,
            bootstrap=args.bootstrap,
            sigmoid_interval=float(act_parts[1]),
            sigmoid_degree=int(act_parts[0]),
        )

    print("\nTraining complete.")

    if args.timing:
        global_timer.print_timing_table()

if __name__ == "__main__":
    main()
