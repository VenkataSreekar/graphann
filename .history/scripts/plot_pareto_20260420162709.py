#!/usr/bin/env python3
"""
plot_pareto.py  —  Plot Recall@10 vs Avg Dist Cmps Pareto curves
================================================================

HOW TO RUN (step by step):
---------------------------
1. Run the benchmarks and SAVE output to files:

    ./scripts/run_sift1m.sh     | tee tmp/results_vamana.txt
    ./scripts/run_sift1m_rlg.sh | tee tmp/results_rlg.txt
    ./scripts/run_sift1m_ivrg.sh| tee tmp/results_ivrg.txt

2. Run this script with those files:

    python3 scripts/plot_pareto.py tmp/results_vamana.txt tmp/results_rlg.txt tmp/results_ivrg.txt

    Or with just two:
    python3 scripts/plot_pareto.py tmp/results_vamana.txt tmp/results_ivrg.txt

3. Plots are saved to:   tmp/plots/

WHAT THE PLOTS SHOW:
--------------------
  pareto_recall_vs_cmps.png   — Recall@10 vs Avg Dist Cmps
                                 (hardware-independent, the key paper figure)
  pareto_recall_vs_latency.png — Recall@10 vs Avg Latency (us)
  nprobe_sweep.png             — IVRG: effect of nprobe (if present in results)
  m_sweep.png                  — RLG:  effect of shell multiplier m

The CORRECT way to read these: a curve that is to the UPPER-RIGHT is better
(higher recall at fewer distance computations / lower latency).
"""

import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.color": "#EEEEEE",
    "grid.linewidth": 0.8,
})

COLORS = {
    "vamana": "#E63946",
    "rlg":    "#2A9D8F",
    "ivrg":   "#457B9D",
    "np1":    "#264653",
    "np2":    "#457B9D",
    "np3":    "#2A9D8F",
    "np5":    "#E9C46A",
    "np8":    "#F4A261",
    "m1.5":   "#457B9D",
    "m2.0":   "#2A9D8F",
    "m2.5":   "#E9C46A",
    "m3.0":   "#F4A261",
}

# ── Parser ────────────────────────────────────────────────────────────────────

def parse_file(path):
    """
    Parse a results file produced by search_index / search_rlg / search_ivrg.
    All three have the SAME output format:

        === Search Results (K=10) ===
               L     Recall@10   Avg Dist Cmps  Avg Latency (us)  P99 Latency (us)
        --------------------------------------------------------------------------
              10        0.7948           484.9             445.6            4749.0

    Returns a dict mapping section_label -> list of (L, recall, cmps, lat, p99).
    """
    sections = {}
    current_label = None
    current_rows  = []

    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()

        # ── Section header detection ──────────────────────────────────────────
        if "Step 3" in line and "RLG" in line:
            current_label = "rlg_main"; current_rows = []
        elif "Step 4" in line and "RLG" in line:
            current_label = "rlg_main"; current_rows = []
        elif "Step 3" in line and ("IVRG" in line or "Searching" in line):
            current_label = "ivrg_main"; current_rows = []
        elif "Step 4" in line and "Vamana" in line:
            current_label = "vamana_main"; current_rows = []
        elif "Step 4" in line and ("Baseline" in line or "baseline" in line):
            current_label = "vamana_main"; current_rows = []
        elif "Step 5" in line and "Vamana" in line:
            current_label = "vamana_main"; current_rows = []
        elif "Step 5" in line and "nprobe" in line.lower():
            current_label = None  # handled below
        elif "nprobe = " in stripped and "---" in stripped:
            np_val = re.search(r'nprobe = (\d+)', stripped)
            if np_val: current_label = f"np{np_val.group(1)}"; current_rows = []
        elif "m = " in stripped and "---" in stripped:
            m_val = re.search(r'm = ([0-9.]+)', stripped)
            if m_val: current_label = f"m{m_val.group(1)}"; current_rows = []
        elif "=== Search Results" in line:
            # Detect "main" section if no header was set
            if current_label is None:
                current_label = "vamana_main"
            current_rows = []
        elif current_label is not None:
            # Try to parse a data row: 5 numbers
            parts = stripped.split()
            if len(parts) >= 5:
                try:
                    L      = int(parts[0])
                    recall = float(parts[1])
                    cmps   = float(parts[2])
                    lat    = float(parts[3])
                    p99    = float(parts[4])
                    current_rows.append((L, recall, cmps, lat, p99))
                    sections[current_label] = list(current_rows)
                except ValueError:
                    pass

    return sections


def detect_label(path, sections):
    """Guess a human-readable label for the results file."""
    name = os.path.basename(path).lower()
    if "ivrg" in name:   return "IVRG (ours)"
    if "rlg"  in name:   return "RLG (ours)"
    if "vamana" in name: return "Vamana (baseline)"
    # Fallback: look for ivrg_main or rlg_main in sections
    if "ivrg_main" in sections: return "IVRG (ours)"
    if "rlg_main"  in sections: return "RLG (ours)"
    return "Vamana (baseline)"


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_pareto_ax(ax, rows, label, color, marker='o'):
    """Plot one Pareto curve (recall vs cmps or latency)."""
    recalls = [r[1] for r in rows]
    cmps    = [r[2] for r in rows]
    ax.plot(cmps, recalls, f'{marker}-', color=color, lw=2, ms=6, label=label)
    # Annotate L values
    for L, rec, cmp, *_ in rows:
        ax.annotate(str(L), (cmp, rec), textcoords="offset points",
                    xytext=(4, 2), fontsize=6, color=color, alpha=0.7)


def finalize_ax(ax, xlabel, ylabel="Recall@10", title=""):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title: ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNo files provided. Looking in tmp/ for result files...")
        candidates = []
        tmp_dir = "tmp"
        for name in ["results_vamana.txt", "results_rlg.txt", "results_ivrg.txt"]:
            p = os.path.join(tmp_dir, name)
            if os.path.exists(p):
                candidates.append(p)
        if not candidates:
            print("  None found. Run one of the benchmark scripts first.")
            print("  Example: ./scripts/run_sift1m.sh | tee tmp/results_vamana.txt")
            sys.exit(1)
        print(f"  Found: {candidates}")
        files = candidates
    else:
        files = sys.argv[1:]

    out_dir = "tmp/plots"

    # ── Parse all files ───────────────────────────────────────────────────────
    all_sections = {}   # filename -> {label: rows}
    all_labels   = {}   # filename -> human label

    for path in files:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping.")
            continue
        secs = parse_file(path)
        all_sections[path] = secs
        all_labels[path]   = detect_label(path, secs)
        print(f"  Parsed {path}: {list(secs.keys())}")

    # ── Figure 1: Recall@10 vs Avg Dist Cmps  (main paper figure) ───────────
    # Collect "main" curves from each file
    main_curves = []  # (human_label, color_key, rows)
    color_map = {"Vamana (baseline)": "vamana",
                 "RLG (ours)":        "rlg",
                 "IVRG (ours)":       "ivrg"}

    for path, secs in all_sections.items():
        human = all_labels[path]
        ck = color_map.get(human, "ivrg")
        for key in ("vamana_main", "rlg_main", "ivrg_main"):
            if key in secs:
                main_curves.append((human, ck, secs[key]))
                break
        else:
            # If only one section exists, use it
            if secs:
                first_key = list(secs.keys())[0]
                main_curves.append((human, ck, secs[first_key]))

    if main_curves:
        markers = ['o', 's', '^', 'D']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Recall vs Avg Dist Cmps
        ax = axes[0]
        for i, (label, ck, rows) in enumerate(main_curves):
            plot_pareto_ax(ax, rows, label, COLORS.get(ck, "#888888"), markers[i % len(markers)])
        finalize_ax(ax, "Avg Distance Computations",
                    title="Recall@10 vs Dist Cmps\n(hardware-independent)")

        # Right: Recall vs Avg Latency
        ax = axes[1]
        for i, (label, ck, rows) in enumerate(main_curves):
            recalls = [r[1] for r in rows]
            lats    = [r[3] for r in rows]
            color   = COLORS.get(ck, "#888888")
            ax.plot(lats, recalls, f'{markers[i % len(markers)]}-',
                    color=color, lw=2, ms=6, label=label)
        finalize_ax(ax, "Avg Latency (µs)",
                    title="Recall@10 vs Latency")

        fig.suptitle("SIFT1M  —  R=32, L_build=75, alpha=1.2",
                     fontsize=13, y=1.01)
        save_fig(fig, f"{out_dir}/pareto_main.png")

    # ── Figure 2: nprobe sweep ────────────────────────────────────────────────
    np_curves = []
    for path, secs in all_sections.items():
        for k, rows in secs.items():
            if k.startswith("np"):
                np_val = k[2:]
                np_curves.append((f"nprobe={np_val}", f"np{np_val}", rows))

    if np_curves:
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, ck, rows in sorted(np_curves, key=lambda x: x[0]):
            plot_pareto_ax(ax, rows, label, COLORS.get(ck, "#888888"))
        # Add Vamana baseline if available
        for path, secs in all_sections.items():
            if "vamana_main" in secs:
                plot_pareto_ax(ax, secs["vamana_main"], "Vamana", COLORS["vamana"], 'D')
                break
        finalize_ax(ax, "Avg Distance Computations",
                    title="IVRG: Effect of nprobe (K=512)\nMore seeds = better routing, diminishing returns")
        save_fig(fig, f"{out_dir}/nprobe_sweep.png")

    # ── Figure 3: m sweep (RLG shell multiplier) ─────────────────────────────
    m_curves = []
    for path, secs in all_sections.items():
        for k, rows in secs.items():
            if k.startswith("m") and re.match(r'm[0-9.]+$', k):
                m_val = k[1:]
                m_curves.append((f"m={m_val}", f"m{m_val}", rows))

    if m_curves:
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, ck, rows in sorted(m_curves, key=lambda x: x[0]):
            plot_pareto_ax(ax, rows, label, COLORS.get(ck, "#888888"))
        for path, secs in all_sections.items():
            if "vamana_main" in secs:
                plot_pareto_ax(ax, secs["vamana_main"], "Vamana", COLORS["vamana"], 'D')
                break
        finalize_ax(ax, "Avg Distance Computations",
                    title="RLG: Effect of Shell Multiplier m\nHigher m = wider shells = more long-range edges")
        save_fig(fig, f"{out_dir}/m_sweep.png")

    print(f"\nAll plots saved to {out_dir}/")
    print("\nRead the plots: a curve that is UPPER-RIGHT is better")
    print("(higher recall at fewer distance computations).")


if __name__ == "__main__":
    main()