import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "e2e_out")
rs = [json.loads(l) for l in open(out / "eval.jsonl")]
ep = [r["epoch"] for r in rs]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for i in range(2):
    a1.semilogy(ep, [r["max_dw"][i] / r["w_scale"][i] for r in rs], "o-", ms=3, label=f"layer {i+1}")
a1.set(xlabel="epoch", ylabel=r"$\max|W_{\mathrm{enc}} - W_{\mathrm{plain}}|\ /\ \max|W_{\mathrm{plain}}|$", title="(a) Weight difference")
a1.set_yticks([5e-8, 1e-7, 2e-7, 5e-7], ["5e-8", "1e-7", "2e-7", "5e-7"], minor=False)
a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a1.legend()
a2.plot(ep, [r["c_plain"] for r in rs], "-", label="plaintext")
a2.plot(ep, [r["c_he_weights"] for r in rs], "o", ms=3, label="encrypted training, plaintext inference")
enc = [r for r in rs if "c_encrypted" in r]
a2.plot([r["epoch"] for r in enc], [r["c_encrypted"] for r in enc], "s", ms=7, mfc="none", label="encrypted training and inference")
a2.set(xlabel="epoch", ylabel="test c-index", title="(b) Test c-index")
a2.legend()
fig.tight_layout(); fig.savefig(out / "e2e_gap.png", dpi=130); fig.savefig(out / "e2e_gap.pdf")
print(out / "e2e_gap.png")
