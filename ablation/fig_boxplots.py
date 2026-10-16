import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ABLATION_LOGS = Path(__file__).parent / "logs"
OUT_DIR = Path(__file__).parent / "figures"
MUTED = '#8a8980'
COL_BASELINE = ['#808080']
COL_CC = ['#9cc3dc', '#3a7ca5', '#1d4560']
COL_PH = ['#e6a3a5', '#c44e52', '#6e2427']
CONFIGS = [
    ("CoxPH 4L\npolynomial", "boxplot_coxph4_k{}.csv", COL_PH, False),
    ("CoxPH 4L\nexact", "boxplot_coxph4_exact_k{}.csv", COL_PH, True),
    ("CoxCC 2L\npolynomial", "boxplot_coxcc2_k{}.csv", COL_CC, False),
    ("CoxCC 2L\nexact", "boxplot_coxcc2_exact_k{}.csv", COL_CC, True),
]
K_VALUES = [1, 10, 50]


def load_csv(path, col="c_index"):
    with open(path) as f:
        return [float(row[col]) for row in csv.DictReader(f)]


def draw(ax, pos, values, colors, hollow):
    bp = ax.boxplot(values, positions=pos, widths=0.62, patch_artist=True, showfliers=True,
                    medianprops=dict(color='white', lw=1.6), flierprops=dict(ms=3, markeredgecolor=MUTED),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for box, med, c in zip(bp['boxes'], bp['medians'], colors):
        if hollow:
            box.set(facecolor='white', edgecolor=c, lw=1.6)
            med.set(color=c)
        else:
            box.set(facecolor=c, edgecolor=c)
    for p, v in zip(pos, values):
        ax.text(p, 0.775, f'{100 * np.median(v):.1f}', ha='center', va='bottom', rotation=90, fontsize=10)


def main():
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED, 'font.size': 9,
                         'axes.titlelocation': 'left'})
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    draw(ax, [-2], [load_csv(ABLATION_LOGS / "baseline_result.csv")], COL_BASELINE, False)
    for i, (_, template, colors, hollow) in enumerate(CONFIGS):
        draw(ax, [4 * i + j for j in range(3)], [load_csv(ABLATION_LOGS / template.format(k)) for k in K_VALUES],
             colors, hollow)
    ax.set_xticks([-2] + [4 * i + j for i in range(len(CONFIGS)) for j in range(3)])
    ax.set_xticklabels(['1'] + [str(k) for k in K_VALUES] * len(CONFIGS), fontsize=10)
    tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (label, *_) in enumerate(CONFIGS):
        ax.text(4 * i + 1, -0.17, label, transform=tr, ha='center', va='top', fontsize=10)
    ax.text(-2, -0.17, 'PiDeeL\nbaseline', transform=tr, ha='center', va='top', fontsize=10)
    ax.text(-2.9, -0.05, '$K$', transform=tr, ha='right', va='top', fontsize=10)
    ax.set_ylabel('test c-index', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylim(0.55, 0.81)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Median (%) above each box', fontsize=11)
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "boxplot_federated.pdf", bbox_inches='tight')
    fig.savefig(OUT_DIR / "boxplot_federated.png", dpi=200, bbox_inches='tight')
    print(f"Saved {OUT_DIR / 'boxplot_federated'}.pdf/.png")


if __name__ == "__main__":
    main()
