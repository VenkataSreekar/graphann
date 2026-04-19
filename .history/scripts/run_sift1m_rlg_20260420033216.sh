#!/usr/bin/env bash
#
# Radial Layer Graph benchmark on SIFT1M — mirrors run_sift1m.sh exactly.
# Usage: ./scripts/run_sift1m_rlg.sh
#
# Prerequisites: run ./scripts/run_sift1m.sh once first so that
#   tmp/sift_base.fbin, tmp/sift_query.fbin, tmp/sift_gt.ibin exist.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/tmp"
BUILD_DIR="$ROOT/build"

BASE_FBIN="$DATA_DIR/sift_base.fbin"
QUERY_FBIN="$DATA_DIR/sift_query.fbin"
GT_IBIN="$DATA_DIR/sift_gt.ibin"
RLG_INDEX="$DATA_DIR/sift_rlg_index.bin"

# ─── 0. Pre-flight check ──────────────────────────────────────────────────────
echo "=== Pre-flight checks ==="
for f in "$BASE_FBIN" "$QUERY_FBIN" "$GT_IBIN"; do
    [ -f "$f" ] || { echo "ERROR: $f not found. Run ./scripts/run_sift1m.sh first."; exit 1; }
done
echo "Data files OK."
echo ""

# ─── 1. Build (compiles build_rlg + search_rlg alongside existing targets) ───
echo "=== Step 1: Building project (including RLG targets) ==="
mkdir -p "$BUILD_DIR"
pushd "$BUILD_DIR" > /dev/null
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
popd > /dev/null
echo ""

# ─── 2. Confirm data files ────────────────────────────────────────────────────
echo "=== Step 2: Using existing SIFT1M binary files ==="
echo "  $BASE_FBIN"
echo "  $QUERY_FBIN"
echo "  $GT_IBIN"
echo ""

# ─── 3. Build RLG index ───────────────────────────────────────────────────────
echo "=== Step 3: Building Radial Layer Graph index ==="
echo "  R=32  L=75  alpha=0.5  m=2.0"
"$BUILD_DIR/build_rlg"  \
    --data   "$BASE_FBIN"  \
    --output "$RLG_INDEX"  \
    --R 32 --L 75 --alpha 0.5 --m 2.0 \
    --two_pass 1
echo ""

# ─── 4. Search with RLG (main result) ────────────────────────────────────────
echo "=== Step 4: Searching with RLG index ==="
"$BUILD_DIR/search_rlg" \
    --index   "$RLG_INDEX"  \
    --data    "$BASE_FBIN"  \
    --queries "$QUERY_FBIN" \
    --gt      "$GT_IBIN"    \
    --K 10                  \
    --L 10,20,30,50,75,100,150,200
echo ""




# echo "=== Done! ==="
# echo ""
# echo "Capture results:  ./scripts/run_sift1m_rlg.sh | tee tmp/results_rlg.txt"
# echo "Plot:             python3 scripts/plot_pareto.py tmp/results_rlg.txt"