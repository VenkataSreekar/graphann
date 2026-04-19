#!/usr/bin/env python3
"""
plot_pareto.py  —  Parse run_sift1m_rlg.sh output and plot Pareto curves.

Usage:
    ./scripts/run_sift1m_rlg.sh | tee tmp/results_rlg.txt
    python3 scripts/plot_pareto.py tmp/results_rlg.txt

Produces:
    tmp/pareto_rlg_vs_vamana.png   — main recall@10 vs QPS comparison
    tmp/sweep_m.png                — effect of m on the Pareto curve
    tmp/sweep_alpha.png            — effect of alpha on the Pareto curve
"""

import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

C = {
    "vamana": "#E63946",
    "rlg":    "#2A9D8F",
    "m1.5":   "#457B9D",
    "m2.0":   "#2A9D8F",
    "m2.5":   "#E9C46A",
    "m3.0":   "#F4A261",
    "a0.0":   "#264653",
    "a0.3":   "#457B9D",
    "a0.5":   "#2A9D8F",
    "a0.7":   "#E9C46A",
    "a0.9":   "#E76F51",
}

# ── parser ────────────────────────────────────────────────────────────────────
# The search binaries print rows like:
#   L    Recall@K   QPS   ...
# with a dashed separator line underneath the header.

def parse_table(lines):
    """
    Given a list of lines from a search binary's output,
    return list of (L, recall, qps) tuples.
    """
    data = []
    for line in lines:
        line = line.strip()
        parts = line.split()
        if len(parts) >= 3:
            try:
                L      = int(parts[0])
                recall = float(parts[1])
                qps    = float(parts[2])
                data.append((recall, qps))
            except ValueError:
                pass
    return data


def parse_results(path):
    with open(path) as f:
        lines = f.readlines()

    sections = {}
    current_label = None
    current_lines = []

    for line in lines:
        # Detect section headers
        if "Step 4" in line or "Searching with RLG index" in line:
            current_label = "rlg_main"
            current_lines = []
        elif "Step 5" in line or "Searching with Vamana" in line:
            current_label = "vamana_main"
            current_lines = []
        elif "m = " in line and "---" in line:
            m_val = re.search(r'm = ([0-9.]+)', line).group(1)
            current_label = f"m{m_val}"
            current_lines = []
        elif "alpha = " in line and "---" in line:
            a_val = re.search(r'alpha = ([0-9.]+)', line).group(1)
            current_label = f"a{a_val}"
            current_lines = []
        elif current_label:
            current_lines.append(line)
            sections[current_label] = list(current_lines)

    parsed = {}
    for k, v in sections.items():
        parsed[k] = parse_table(v)
    return parsed


# ── plot helpers ──────────────────────────────────────────────────────────────

def plot_two_curves(ax, data1, data2, label1, label2, c1, c2, K=10):
    if data1:
        r1 = [r for r,_ in data1]; q1 = [q for _,q in data1]
        ax.plot(r1, q1, 'o-', color=c1, lw=2, ms=6, label=label1)
    if data2:
        r2 = [r for r,_ in data2]; q2 = [q for _,q in data2]
        ax.plot(r2, q2, 's-', color=c2, lw=2, ms=6, label=label2)
    ax.set_xlabel(f"Recall@{K}", fontsize=11)
    ax.set_ylabel("QPS", fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    ax.grid(True, color="#EEEEEE", linewidth=0.8)


def save(fig, path):
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/plot_pareto.py <results_rlg.txt>")
        sys.exit(1)

    results_path = sys.argv[1]
    out_dir = os.path.dirname(results_path) or "tmp"

    try:
        sections = parse_results(results_path)
    except FileNotFoundError:
        print(f"File not found: {results_path}")
        sys.exit(1)

    K = 10

    # ── 1. Main comparison: RLG vs Vamana ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_two_curves(ax,
        sections.get("vamana_main", []),
        sections.get("rlg_main",    []),
        "Vamana (DiskANN)",
        "Radial Layer Graph (ours)",
        C["vamana"], C["rlg"], K)
    ax.set_title("Recall@10 vs QPS — Vamana vs RLG\nSIFT1M, R=32, L_build=75",
                 fontsize=12, fontweight="bold")
    save(fig, f"{out_dir}/pareto_rlg_vs_vamana.png")

    # ── 2. m sweep ────────────────────────────────────────────────────────────
    m_keys = [k for k in sections if k.startswith("m")]
    if m_keys:
        fig, ax = plt.subplots(figsize=(7, 5))
        for mk in sorted(m_keys):
            m_val = mk[1:]
            color = C.get(mk, "#888888")
            data  = sections[mk]
            if data:
                r = [x for x,_ in data]; q = [x for _,x in data]
                ax.plot(r, q, 'o-', color=color, lw=2, ms=5, label=f"m={m_val}")
        ax.set_xlabel(f"Recall@{K}", fontsize=11)
        ax.set_ylabel("QPS", fontsize=11)
        ax.set_title("Effect of Shell Multiplier m\nRLG on SIFT1M, R=32, L_build=75",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1.02)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        ax.grid(True, color="#EEEEEE", linewidth=0.8)
        save(fig, f"{out_dir}/sweep_m.png")

    # ── 3. alpha sweep ────────────────────────────────────────────────────────
    a_keys = [k for k in sections if k.startswith("a")]
    if a_keys:
        fig, ax = plt.subplots(figsize=(7, 5))
        for ak in sorted(a_keys):
            a_val = ak[1:]
            color = C.get(ak, "#888888")
            data  = sections[ak]
            if data:
                r = [x for x,_ in data]; q = [x for _,x in data]
                ax.plot(r, q, 's-', color=color, lw=2, ms=5, label=f"α={a_val}")
        ax.set_xlabel(f"Recall@{K}", fontsize=11)
        ax.set_ylabel("QPS", fontsize=11)
        ax.set_title("Effect of Angular Diversity Threshold α\nRLG on SIFT1M, R=32, L_build=75",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1.02)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
        ax.grid(True, color="#EEEEEE", linewidth=0.8)
        save(fig, f"{out_dir}/sweep_alpha.png")

    print("\nAll plots saved. Use these in your report.")


if __name__ == "__main__":
    main()
