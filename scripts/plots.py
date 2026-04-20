#!/usr/bin/env python3
"""
compare_all.py — Run all 4 ANNS implementations on SIFT1M and produce a
                 side-by-side recall vs latency comparison.

The script:
  1. Builds the project (cmake + make).
  2. Runs all 4 search binaries at each L value.
  3. Parses stdout tables into a unified DataFrame.
  4. Prints a formatted comparison table.
  5. Saves results to CSV in the results/ folder.
  6. Plots Recall vs Latency and Recall vs Dist-Cmps in the results/ folder.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ── Optional imports ─────────────────────────────────────────────────────────
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[warn] pandas not found — CSV output disabled. Install with: pip install pandas")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not found — plots disabled. Install with: pip install matplotlib")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration: one entry per implementation
# ─────────────────────────────────────────────────────────────────────────────
IMPLEMENTATIONS = [
    {
        "name":       "Vamana",
        "build_bin":  "build_index",
        "search_bin": "search_index",
        "index_file": lambda cfg: cfg["tmp"] / "sift_index.bin",
        "build_args": lambda cfg: [
            "--data",   str(cfg["base"]),
            "--output", str(cfg["tmp"] / "sift_index.bin"),
            "--R",      "32",
            "--L",      "75",
            "--alpha",  "1.2",
            "--gamma",  "1.5",
        ],
        "search_args": lambda cfg: [
            "--index",   str(cfg["tmp"] / "sift_index.bin"),
            "--data",    str(cfg["base"]),
            "--queries", str(cfg["query"]),
            "--gt",      str(cfg["gt"]),
            "--K",       str(cfg["K"]),
            "--L",       cfg["L_str"],
        ],
    },
    {
        "name":       "IVRG",
        "build_bin":  "build_ivrg",
        "search_bin": "search_ivrg",
        "index_file": lambda cfg: cfg["tmp"] / "sift_ivrg_index.bin",
        "build_args": lambda cfg: [
            "--data",   str(cfg["base"]),
            "--output", str(cfg["tmp"] / "sift_ivrg_index.bin"),
            "--R",      "32",
            "--L",      "75",
            "--alpha",  "1.2",
            "--gamma",  "1.5",
            "--K",      "512",
            "--nprobe", "3",
            "--T",      "15",
        ],
        "search_args": lambda cfg: [
            "--index",   str(cfg["tmp"] / "sift_ivrg_index.bin"),
            "--data",    str(cfg["base"]),
            "--queries", str(cfg["query"]),
            "--gt",      str(cfg["gt"]),
            "--K",       str(cfg["K"]),
            "--L",       cfg["L_str"],
        ],
    },
    {
        "name":       "RLG",
        "build_bin":  "build_rlg",
        "search_bin": "search_rlg",
        "index_file": lambda cfg: cfg["tmp"] / "sift_rlg_index.bin",
        "build_args": lambda cfg: [
            "--data",   str(cfg["base"]),
            "--output", str(cfg["tmp"] / "sift_rlg_index.bin"),
            "--R",      "32",
            "--L",      "75",
            "--alpha",  "1.2",
            "--gamma",  "1.5",
            "--m",      "2.0",
        ],
        "search_args": lambda cfg: [
            "--index",   str(cfg["tmp"] / "sift_rlg_index.bin"),
            "--data",    str(cfg["base"]),
            "--queries", str(cfg["query"]),
            "--gt",      str(cfg["gt"]),
            "--K",       str(cfg["K"]),
            "--L",       cfg["L_str"],
        ],
    },
    {
        "name":       "Vamana-JL",
        "build_bin":  "build_index_jl",
        "search_bin": "search_index_jl",
        "index_file": lambda cfg: cfg["tmp"] / "sift1m_jl.bin",
        "build_args": lambda cfg: [
            "--data",     str(cfg["base"]),
            "--output",   str(cfg["tmp"] / "sift1m_jl.bin"),
            "--R",        "32",
            "--L_build",  "200",
            "--alpha",    "1.2",
            "--gamma",    "1.5",
            "--proj_dim", str(cfg["proj_dim"]),
        ],
        "search_args": lambda cfg: [
            "--index",   str(cfg["tmp"] / "sift1m_jl.bin"),
            "--data",    str(cfg["base"]),
            "--queries", str(cfg["query"]),
            "--gt",      str(cfg["gt"]),
            "--K",       str(cfg["K"]),
            "--L",       cfg["L_str"],
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd, label=""):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=False)
    if result.returncode != 0:
        print(f"[error] {label} exited with code {result.returncode}", file=sys.stderr)
        sys.exit(1)


def run_capture(cmd):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")
    return result.stdout


def parse_results_table(output: str, impl_name: str) -> list[dict]:
    rows = []
    in_table = False
    for line in output.splitlines():
        line = line.strip()
        if re.match(r"^-{30,}$", line):
            in_table = True
            continue
        if not in_table: continue
        if not line or line.startswith("Done") or line.startswith("==="):
            in_table = False
            continue
        parts = line.split()
        if len(parts) < 5: continue
        try:
            rows.append({
                "impl":          impl_name,
                "L":             int(parts[0]),
                "recall":        float(parts[1]),
                "avg_dist_cmps": float(parts[2]),
                "avg_lat_us":    float(parts[3]),
                "p99_lat_us":    float(parts[4]),
            })
        except ValueError: continue
    return rows


def preflight(cfg):
    missing = []
    for f in [cfg["base"], cfg["query"], cfg["gt"]]:
        if not f.exists():
            missing.append(str(f))
    if missing:
        print("[error] Required data files not found in tmp/:")
        for m in missing:
            print(f"  {m}")
        print("Run ./scripts/run_sift1m.sh first to download and convert SIFT1M.")
        sys.exit(1)


def print_comparison_table(all_rows: list[dict], K: int):
    impls = list(dict.fromkeys(r["impl"] for r in all_rows))
    L_vals = sorted(set(r["L"] for r in all_rows))
    idx = {(r["impl"], r["L"]): r for r in all_rows}
    col_w, name_w = 14, 12

    def hr(): print("-" * (name_w + len(impls) * col_w * 2 + len(impls) * 2 + 4))

    print(f"\n{'':>{name_w}}", end="")
    for impl in impls: print(f"  {impl:^{col_w * 2}}", end="")
    print()

    print(f"{'L':>{name_w}}", end="")
    for _ in impls: print(f"  {'Recall@'+str(K):>{col_w}}{'AvgLat(us)':>{col_w}}", end="")
    print()
    hr()

    for L in L_vals:
        print(f"{L:>{name_w}}", end="")
        for impl in impls:
            row = idx.get((impl, L))
            if row: print(f"  {row['recall']:>{col_w}.4f}{row['avg_lat_us']:>{col_w}.1f}", end="")
            else: print(f"  {'N/A':>{col_w}}{'N/A':>{col_w}}", end="")
        print()
    hr()


def plot_results(all_rows: list[dict], results_dir: Path, K: int):
    """Generate recall vs latency and recall vs dist-cmps plots using Matplotlib."""
    if not HAS_MPL: return
    impls = list(dict.fromkeys(r["impl"] for r in all_rows))
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    markers = ["o", "s", "^", "D"]

    for metric, ylabel, filename in [
        ("avg_lat_us",    "Avg Latency (µs)",         "compare_recall_latency.png"),
        ("avg_dist_cmps", "Avg Distance Computations", "compare_recall_distcmps.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, impl in enumerate(impls):
            rows = sorted([r for r in all_rows if r["impl"] == impl], key=lambda r: r["L"])
            if not rows: continue
            x = [r["recall"] for r in rows]
            y = [r[metric] for r in rows]
            L = [r["L"] for r in rows]
            ax.plot(x, y, color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    linewidth=2, markersize=7, label=impl)
            for xi, yi, li in zip(x, y, L):
                ax.annotate(f"L={li}", (xi, yi), textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=colors[i % len(colors)])

        ax.set_xlabel(f"Recall@{K}", fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Recall vs {ylabel} — SIFT1M (K={K})", fontsize=13)
        ax.legend(fontsize=11); ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out = results_dir / filename
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  Plot saved: {out}")


def save_csv(all_rows: list[dict], path: Path):
    if not HAS_PANDAS:
        if not all_rows: return
        keys = list(all_rows[0].keys())
        with open(path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in all_rows: f.write(",".join(str(row[k]) for k in keys) + "\n")
    else:
        pd.DataFrame(all_rows).to_csv(path, index=False)
    print(f"  CSV saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    
    # Define and create the results directory
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(description="Compare all 4 ANNS implementations on SIFT1M.")
    ap.add_argument("--root",       type=Path, default=root)
    ap.add_argument("--tmp",        type=Path, default=None)
    ap.add_argument("--K",          type=int,  default=10)
    ap.add_argument("--L",          type=str,  default="10,20,30,50,75,100,150,200")
    ap.add_argument("--proj_dim",   type=int,  default=100)
    ap.add_argument("--skip_build", action="store_true")
    ap.add_argument("--skip_index", action="store_true")
    ap.add_argument("--output",     type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    tmp = args.tmp.resolve() if args.tmp else root / "tmp"
    build_dir = root / "build"

    cfg = {
        "root": root, "tmp": tmp, "build": build_dir,
        "base": tmp / "sift_base.fbin", "query": tmp / "sift_query.fbin", "gt": tmp / "sift_gt.ibin",
        "K": args.K, "L_str": args.L, "proj_dim": args.proj_dim,
    }
    out_csv = args.output or results_dir / "compare_results.csv"

    print("=== Pre-flight: checking data files ===")
    preflight(cfg)
    print("  Data files OK.\n")

    if not args.skip_build:
        print("=== Step 1: Building project ===")
        run(["cmake", "-S", str(root), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"])
        run(["cmake", "--build", str(build_dir), "--parallel"])
        print()

    if not args.skip_index:
        print("=== Step 2: Building indices ===")
        for impl in IMPLEMENTATIONS:
            print(f"\n  [{impl['name']}] Building index...")
            bin_path = build_dir / impl["build_bin"]
            run([str(bin_path)] + impl["build_args"](cfg), label=impl["name"])
        print()

    print("=== Step 3: Running search across all implementations ===")
    all_rows = []
    for impl in IMPLEMENTATIONS:
        print(f"\n  [{impl['name']}] Searching...")
        bin_path = build_dir / impl["search_bin"]
        try:
            output = run_capture([str(bin_path)] + impl["search_args"](cfg))
            rows = parse_results_table(output, impl["name"])
            if not rows: print(f"  [warn] No rows parsed from {impl['name']}.")
            else: 
                print(f"  Parsed {len(rows)} L-value rows.")
                all_rows.extend(rows)
        except RuntimeError as e: print(f"  [error] {e} — skipping {impl['name']}")

    if not all_rows:
        print("\n[error] No results parsed. Exiting."); sys.exit(1)

    print("\n" + "=" * 80)
    print(f"  COMPARISON RESULTS  —  SIFT1M  K={cfg['K']}")
    print("=" * 80)
    print_comparison_table(all_rows, cfg["K"])

    print("=== Step 4: Saving results ===")
    save_csv(all_rows, out_csv)

    print("=== Step 5: Plotting ===")
    plot_results(all_rows, results_dir, cfg["K"])

    print("\n=== Done! ===")
    print(f"  Results CSV  : {out_csv}")
    print(f"  Latency plot : {results_dir / 'compare_recall_latency.png'}")
    print(f"  DistCmp plot : {results_dir / 'compare_recall_distcmps.png'}")

if __name__ == "__main__":
    main()