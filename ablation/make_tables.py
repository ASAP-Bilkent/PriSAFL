import csv
import statistics
from pathlib import Path

LOGS = Path(__file__).parent / "logs"

ABLATION = [
    ("Baseline, CoxPH 4-layer", "baseline_result.csv"),
    ("HE-ready CoxPH 4-layer", "boxplot_coxph4_k{k}.csv"),
    ("HE-ready CoxPH 2-layer", "boxplot_coxph2_k{k}.csv"),
    ("HE-ready CoxCC 4-layer", "boxplot_coxcc4_k{k}.csv"),
    ("Pri-SAFL (HE-ready CoxCC 2-layer)", "boxplot_coxcc2_k{k}.csv"),
    ("Plaintext exact CoxPH 4-layer", "boxplot_coxph4_exact_k{k}.csv"),
    ("Plaintext exact CoxCC 2-layer", "boxplot_coxcc2_exact_k{k}.csv"),
]

PATIENT = [
    ("Baseline, CoxPH 4-layer", "baseline_ver.csv"),
    ("HE-ready CoxCC 2-layer", "patient_coxcc2_k{k}.csv"),
    ("HE-ready CoxPH 4-layer", "patient_coxph4_k{k}.csv"),
]

NONIID = [("Stratified", "boxplot_{loss}_k{k}.csv")] + [
    (f"beta={b:g}", f"noniid_{{loss}}_b{b}_k{{k}}.csv") for b in (5.0, 1.0, 0.5)]

DEPTH = [
    ("2-layer", "boxplot_coxcc2_k1.csv", "boxplot_coxph2_k1.csv"),
    ("3-layer", "boxplot_coxcc3_k1.csv", "boxplot_coxph3_k1.csv"),
    ("4-layer", "boxplot_coxcc4_k1.csv", "boxplot_coxph4_k1.csv"),
]
ACT_DEGREES = [3, 5, 7, 9]
ACT_INTERVALS = [3, 5, 10, 15]

def c_indices(path):
    with open(path) as f:
        return [float(row["c_index"]) for row in csv.DictReader(f)]

def median_or(path):
    return f"{100 * statistics.median(c_indices(path)):.2f}" if path.exists() else "--"

def median_mean(path):
    if not path.exists():
        return "-- | --"
    values = c_indices(path)
    return f"{100 * statistics.median(values):.2f} | {100 * statistics.mean(values):.2f}"

def k_table(configs, ks):
    header = "| Config | " + " | ".join(f"K={k} median | K={k} mean" for k in ks) + " |"
    rows = [header, "|" + "---|" * (2 * len(ks) + 1)]
    for label, template in configs:
        cells = [median_mean(LOGS / template.format(k=k)) if "{k}" in template or k == 1 else "-- | --"
                 for k in ks]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)

def noniid_table():
    cols = [(k, loss) for k in (10, 50) for loss in ("coxcc2", "coxph4")]
    header = "| Partition | " + " | ".join(f"K={k} {loss} median | K={k} {loss} mean" for k, loss in cols) + " |"
    rows = [header, "|" + "---|" * (2 * len(cols) + 1)]
    for label, template in NONIID:
        cells = [median_mean(LOGS / template.format(loss=loss, k=k)) for k, loss in cols]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)

def depth_table():
    rows = ["| Depth | CoxCC | CoxPH |", "|---|---|---|"]
    for label, cc, ph in DEPTH:
        rows.append(f"| {label} | {median_or(LOGS / cc)} | {median_or(LOGS / ph)} |")
    return "\n".join(rows)

def activation_table():
    header = "| Degree | " + " | ".join(f"I={iv}" for iv in ACT_INTERVALS) + " |"
    rows = [header, "|" + "---|" * (len(ACT_INTERVALS) + 1)]
    for d in ACT_DEGREES:
        cells = [median_or(LOGS / f"act_{d}_{iv}.csv") for iv in ACT_INTERVALS]
        rows.append(f"| {d} | " + " | ".join(cells) + " |")
    return "\n".join(rows)

if __name__ == "__main__":
    print("C-index of the baseline and the HE-ready variants (%, 15 test runs)\n")
    print(k_table(ABLATION, [1, 50]))
    print("\nMedian c-index for polynomial activations fitted to ReLU by the Remez algorithm"
          " (CoxPH 4-layer, lr 0.01, K=1, -- marks divergence)\n")
    print(activation_table())
    print("\nNetwork depth vs. accuracy (median c-index, K=1)\n")
    print(depth_table())
    print("\nC-index with patient-level splits on the 339 verified spectra\n")
    print(k_table(PATIENT, [1, 10, 50]))
    print("\nC-index under label-skewed partitions (Dirichlet beta)\n")
    print(noniid_table())
