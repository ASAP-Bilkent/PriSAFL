import argparse, copy, importlib.util, json, resource, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd, torch
from sklearn.model_selection import KFold
from pycox.evaluation import EvalSurv

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ablation_main", ROOT / "ablation" / "main.py")
am = importlib.util.module_from_spec(spec); spec.loader.exec_module(am)

from he.encryption import CKKSContext
from he.prisafltext import PriSAFLText
from he.model import PriSAFLHE
from he.trainer import train_epoch_he
from he.main import DIMS
from he.tests.test_utils import ModelArgs, decrypt_scalar, set_weights

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=81)
p.add_argument("--fold", type=int, default=2)
p.add_argument("--n-clients", type=int, default=10)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--lr", type=float, default=0.045)
p.add_argument("--momentum", type=float, default=0.9)
p.add_argument("--ring-dim-exp", type=int, default=None)
p.add_argument("--ckpt-every", type=int, default=5)
p.add_argument("--enc-every", type=int, default=25)
p.add_argument("--out", default=str(Path(__file__).parent / "e2e_out"))
a = p.parse_args()
assert a.n_clients > 1, "K=1 takes the EvalBootstrap path, not paired here"
out = Path(a.out); out.mkdir(exist_ok=True)
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
rss_gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

tr, te = list(KFold(5, shuffle=True, random_state=a.seed).split(am.events))[a.fold]
X_train, X_test, y_train, y_test = am._load_fold(tr, te, a.seed)
mask = am.pathway_info_full.astype(float)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

t0 = time.time()
ctx = CKKSContext(n_clients=a.n_clients, ring_dim_exp=a.ring_dim_exp); PriSAFLText.set_ckks_context(ctx)
log(f"setup {time.time()-t0:.0f}s ring 2^{ctx.cc.GetRingDimension().bit_length()-1} depth {ctx.depth} "
    f"maxrss {rss_gb():.1f} GB")
he = PriSAFLHE(DIMS, mask, ModelArgs(momentum=a.momentum, lr=a.lr))

am._seed_everything(a.seed)
net, optim = am._build_model(SimpleNamespace(n_layers=2, lr=a.lr, l2=0.0, momentum=a.momentum),
                             X_train.shape[1], torch.as_tensor(am.pathway_info_full).to(dev), "3-15", dev)
client_data = am.partition_by_events(X_train, y_train, a.n_clients)
am._log_partition(client_data)
crit = am.CoxCCPreSampledLoss(1, approx_sigmoid_fn=am._make_approx_sigmoid("3-15cT4"))
T = lambda x: torch.tensor(x).float().to(dev)
all_f, all_l = T(X_train), T(y_train)

def torch_weights(m):
    return [(m.pathway_info * m.fc1.weight.t()).detach().cpu().double().numpy(), m.fc_out.weight.t().detach().cpu().double().numpy()]
set_weights(he, torch_weights(net))

def slot_positions(layer):
    sm = layer.slot_multipliers[layer.rank - 1]
    step = sm * (layer.padded_in_dim if layer.rank % 2 == 1 else layer.padded_out_dim)
    idx = np.arange(1, layer.padded_in_dim * layer.padded_out_dim + 1, dtype=float).reshape(layer.padded_in_dim, -1)
    t = PriSAFLText(idx); t._pad_data("col" if layer.rank % 2 == 1 else "row", pad_step=step, pad_batch=1)
    vals, first = np.unique(np.ravel(t._padded_data), return_index=True)
    pos = dict(zip(vals, first))
    return np.array([pos[k] for k in idx.ravel()]).reshape(idx.shape)[:layer.in_dim, :layer.out_dim]
POS = [slot_positions(l) for l in he.layers]

def decrypt_weights(texts):
    ws = []
    for t, pos in zip(texts, POS):
        pt = PriSAFLText._decrypt(t.cipher_data); pt.SetLength(int(pos.max()) + 1)
        ws.append(np.array(pt.GetRealPackedValue())[pos])
    ws[0] = ws[0] * mask
    return ws

def cindex(s_tr, s_te):
    dtr, etr = y_train[:, 1], y_train[:, 0]
    u = np.unique(dtr); eg = np.exp(s_tr - s_tr.max())
    H0 = np.cumsum([etr[dtr == t].sum() / eg[dtr >= t].sum() for t in u])
    surv = pd.DataFrame(np.exp(-np.outer(H0, np.exp(s_te - s_tr.max()))), index=u)
    return EvalSurv(surv, y_test[:, 1], y_test[:, 0], censor_surv="km").concordance_td()

def evaluate(epoch, encrypted_scores):
    he.apply_lookahead()
    y_he, x_he = decrypt_weights([l.W for l in he.layers]), decrypt_weights([l._x_main for l in he.layers])
    w_pt = torch_weights(net)
    dw = [float(np.abs(h - t).max()) for h, t in zip(y_he, w_pt)]
    with torch.no_grad():
        s_tr, s_te = net(all_f).cpu().numpy().ravel(), net(T(X_test)).cpu().numpy().ravel()
    c_pt = am._evaluate(net, all_f, all_l, X_test, y_test, dev)
    net_he = copy.deepcopy(net)
    with torch.no_grad():
        net_he.fc1.weight.copy_(torch.tensor(y_he[0].T)); net_he.fc_out.weight.copy_(torch.tensor(y_he[1].T))
    rec = dict(epoch=epoch, c_plain=c_pt, c_he_weights=am._evaluate(net_he, all_f, all_l, X_test, y_test, dev),
               c_plain_helper=cindex(s_tr, s_te), max_dw=dw, w_scale=[float(np.abs(t).max()) for t in w_pt])
    if encrypted_scores:
        t0 = time.time()
        h_tr = np.array([decrypt_scalar(he.forward(X_train[s:s+1])[-1]) for s in range(len(X_train))])
        h_te = np.array([decrypt_scalar(he.forward(X_test[s:s+1])[-1]) for s in range(len(X_test))])
        rec.update(c_encrypted=cindex(h_tr, h_te), max_dscore_train=float(np.abs(h_tr - s_tr).max()),
                   max_dscore_test=float(np.abs(h_te - s_te).max()), score_scale=float(np.abs(s_tr).max()),
                   inference_s=time.time() - t0)
        np.savez(out / f"scores_e{epoch}.npz", he_tr=h_tr, he_te=h_te, pt_tr=s_tr, pt_te=s_te)
    for l in he.layers:
        l.W = l._x_main
    np.savez(out / f"ckpt_e{epoch}.npz", *x_he, *decrypt_weights([l.W_prev for l in he.layers]))
    torch.save(dict(net=net.state_dict(), optim=optim.state_dict(), np_rng=np.random.get_state()),
               out / f"ckpt_e{epoch}.pt")
    log("EVAL", json.dumps(rec))
    with open(out / "eval.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")

for epoch in range(1, a.epochs + 1):
    t0 = time.time()
    optim.zero_grad()
    he_clients, he_pairs = [], []
    for cx, cy in client_data:
        ev, ctrl, uidx = am.coxcc_sample_pairs(cy[:, 1], cy[:, 0], 1)
        if ev is None:
            he_clients.append((cx[:1], None, None)); he_pairs.append([])
            continue
        crit(net(T(cx[uidx])), torch.tensor(ev, device=dev), torch.tensor(ctrl, device=dev)).backward()
        he_clients.append((cx[uidx], None, None)); he_pairs.append([(int(i), int(j)) for i, j in zip(ev, ctrl)])
    am._scale_grads(net, len(X_train))
    optim.step()
    t_pt = time.time() - t0

    rng = np.random.get_state()
    train_epoch_he(he, X_train, None, None, a.lr, n_clients=a.n_clients, client_data=he_clients,
                   client_pairs=he_pairs, bootstrap=True, sigmoid_interval=15.0, sigmoid_degree=3)
    np.random.set_state(rng)
    log(f"epoch {epoch}/{a.epochs} HE {time.time()-t0-t_pt:.0f}s pairs {sum(map(len, he_pairs))} "
        f"maxrss {rss_gb():.1f} GB")
    stop = (out / "STOP").exists()
    last = stop or epoch == a.epochs
    if epoch % a.ckpt_every == 0 or last:
        evaluate(epoch, encrypted_scores=last or epoch % a.enc_every == 0)
    if stop:
        log("STOP file found, exiting"); break
