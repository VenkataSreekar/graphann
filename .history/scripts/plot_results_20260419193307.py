#!/usr/bin/env python3
"""
plot_results.py  —  Visualize RLG vs Vamana benchmarks

Usage:
  python plot_results.py

Expects two TSV files produced by compare binary's stdout (redirect with tee):
  vamana_results.tsv   :  L  recall  qps
  rlg_results.tsv      :  L  recall  qps  avg_cands

Or pass raw numbers inline (see MANUAL DATA section below).

Also plots degree distribution histogram if degree_dist.tsv is present:
  node_id  degree
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, sys

plt.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── COLORS ──────────────────────────────────────────────────
C_VAMANA = "#E63946"   # vivid red
C_RLG    = "#2A9D8F"   # teal
C_GRID   = "#EEEEEE"

# ============================================================
#  1.  QPS vs Recall@K  Pareto Curve
# ============================================================

def plot_pareto(vamana_data, rlg_data, K=10, out="pareto_curve.png"):
    """
    vamana_data, rlg_data: list of (recall, qps) tuples sorted by recall
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("white")
    ax.grid(True, color=C_GRID, linewidth=0.8)

    v_recall = [r for r,_ in vamana_data]
    v_qps    = [q for _,q in vamana_data]
    r_recall = [r for r,_ in rlg_data]
    r_qps    = [q for _,q in rlg_data]

    ax.plot(v_recall, v_qps, 'o-', color=C_VAMANA, lw=2, ms=6, label="Vamana (DiskANN)")
    ax.plot(r_recall, r_qps, 's-', color=C_RLG,    lw=2, ms=6, label="Radial Layer Graph (ours)")

    # Annotate L values
    for (rec, qps), (L_rec, L_qps) in zip(vamana_data, vamana_data):
        pass  # add annotation if you include L column

    ax.set_xlabel(f"Recall@{K}", fontsize=12)
    ax.set_ylabel("Queries / second (QPS)", fontsize=12)
    ax.set_title(f"Recall@{K} vs. QPS  —  Pareto Frontier", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    ax.set_xlim(0, 1.02)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ============================================================
#  2.  Recall vs Candidates Evaluated  (efficiency plot)
# ============================================================

def plot_efficiency(vamana_data, rlg_data, K=10, out="efficiency.png"):
    """
    vamana_data: list of (recall, avg_candidates)
    rlg_data:    list of (recall, avg_candidates)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("white")
    ax.grid(True, color=C_GRID, linewidth=0.8)

    ax.plot([c for _,c in vamana_data], [r for r,_ in vamana_data],
            'o-', color=C_VAMANA, lw=2, ms=6, label="Vamana")
    ax.plot([c for _,c in rlg_data],    [r for r,_ in rlg_data],
            's-', color=C_RLG,    lw=2, ms=6, label="RLG (ours)")

    ax.set_xlabel("Avg. Distance Computations per Query", fontsize=12)
    ax.set_ylabel(f"Recall@{K}", fontsize=12)
    ax.set_title("Recall vs. Candidates Evaluated\n(hardware-independent efficiency)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ============================================================
#  3.  Degree Distribution Histogram
# ============================================================

def plot_degree_dist(vamana_degrees, rlg_degrees, R, out="degree_dist.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, degs, label, color in zip(
            axes,
            [vamana_degrees, rlg_degrees],
            ["Vamana", "Radial Layer Graph"],
            [C_VAMANA, C_RLG]):
        ax.hist(degs, bins=50, color=color, edgecolor="white", lw=0.5, alpha=0.85)
        mean = np.mean(degs)
        gini = gini_coef(degs)
        ax.axvline(mean, color="black", lw=1.5, ls="--", label=f"mean={mean:.1f}")
        ax.axvline(R,    color="gray",  lw=1,   ls=":",  label=f"R={R}")
        ax.set_xlabel("Node Degree", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{label}\nGini={gini:.3f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_facecolor("white")
        ax.grid(True, axis='y', color=C_GRID)

    fig.suptitle("Degree Distribution Comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def gini_coef(arr):
    a = np.sort(np.array(arr, dtype=float))
    n = len(a)
    return (2 * np.sum((np.arange(1, n+1)) * a) / (n * np.sum(a))) - (n+1)/n


# ============================================================
#  4.  Shell Distribution  (how many nodes per shell, avg)
# ============================================================

def plot_shell_dist(shell_counts_per_node, K=12, out="shell_dist.png"):
    """
    shell_counts_per_node: list of lists — for each node, how many neighbors per shell
    """
    avg_per_shell = np.zeros(K)
    for sc in shell_counts_per_node:
        for k, cnt in enumerate(sc):
            if k < K: avg_per_shell[k] += cnt
    avg_per_shell /= len(shell_counts_per_node)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(K), avg_per_shell, color=C_RLG, edgecolor="white", lw=0.5)
    ax.set_xlabel("Shell index k  (distance band: [mᵏ⁻¹·r, mᵏ·r))", fontsize=11)
    ax.set_ylabel("Avg. neighbors", fontsize=11)
    ax.set_title("Average Neighbors per Radial Shell\n(shows multi-scale connectivity)", fontsize=12)
    ax.set_xticks(range(K))
    ax.set_facecolor("white")
    ax.grid(True, axis='y', color=C_GRID)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ============================================================
#  SAMPLE DATA — replace with your actual numbers
# ============================================================
#  Format: (recall@10, qps)  — copy from compare binary output

VAMANA_SAMPLE = [
    (0.42, 18000),
    (0.62, 12000),
    (0.78,  7800),
    (0.87,  5200),
    (0.93,  3400),
    (0.97,  2200),
    (0.99,  1500),
]

RLG_SAMPLE = [
    (0.45, 17500),
    (0.65, 11800),
    (0.81,  7500),
    (0.89,  5000),
    (0.95,  3200),
    (0.98,  2100),
    (0.995, 1400),
]

# For efficiency plot: (recall, avg_candidates_evaluated)
VAMANA_EFF = [(r, 10 + i*15) for i,(r,_) in enumerate(VAMANA_SAMPLE)]
RLG_EFF    = [(r, 10 + i*12) for i,(r,_) in enumerate(RLG_SAMPLE)]  # RLG evaluates fewer

VAMANA_DEGREES = list(np.random.normal(28, 6, 10000).clip(1, 64).astype(int))
RLG_DEGREES    = list(np.random.normal(26, 4, 10000).clip(1, 64).astype(int))

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    plot_pareto(VAMANA_SAMPLE, RLG_SAMPLE, K=10, out="plots/pareto_curve.png")
    plot_efficiency(VAMANA_EFF, RLG_EFF, K=10, out="plots/efficiency.png")
    plot_degree_dist(VAMANA_DEGREES, RLG_DEGREES, R=32, out="plots/degree_dist.png")
    print("\nAll plots saved to plots/")
    print("Replace VAMANA_SAMPLE / RLG_SAMPLE with your actual benchmark numbers.")
