#!/usr/bin/env bash
#
# IVRG benchmark on SIFT1M.
# Usage: ./scripts/run_sift1m_ivrg.sh | tee tmp/results_ivrg.txt
#
# Prerequisites: run ./scripts/run_sift1m.sh first.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/tmp"
BUILD_DIR="$ROOT/build"

BASE_FBIN="$DATA_DIR/sift_base.fbin"
QUERY_FBIN="$DATA_DIR/sift_query.fbin"
GT_IBIN="$DATA_DIR/sift_gt.ibin"
IVRG_INDEX="$DATA_DIR/sift_ivrg_index.bin"
VAMANA_INDEX="$DATA_DIR/sift_index.bin"

L_LIST="10,20,30,50,75,100,150,200"

# ─── 0. Pre-flight ────────────────────────────────────────────────────────────
echo "=== Pre-flight checks ==="
for f in "$BASE_FBIN" "$QUERY_FBIN" "$GT_IBIN"; do
    [ -f "$f" ] || { echo "ERROR: $f missing. Run ./scripts/run_sift1m.sh first."; exit 1; }
done
echo "Data files OK."
echo ""

# ─── 1. Build ─────────────────────────────────────────────────────────────────
echo "=== Step 1: Building project ==="
mkdir -p "$BUILD_DIR"
pushd "$BUILD_DIR" > /dev/null
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
popd > /dev/null
echo ""

# ─── 2. Build IVRG index ──────────────────────────────────────────────────────
echo "=== Step 2: Building IVRG index ==="
echo "  Vamana graph: R=32 L=75 alpha=1.2 gamma=1.5"
echo "  Routing layer: K=512 nprobe=3 T=15"
echo "  (Representative finding is now O(S*K) — no more O(K*N) 60-second scan)"
"$BUILD_DIR/build_ivrg" \
    --data   "$BASE_FBIN" \
    --output "$IVRG_INDEX" \
    --R 32 --L 75 --alpha 1.2 --gamma 1.5 \
    --K 512 --nprobe 3 --T 15
echo ""

# ─── 3. Search with IVRG  (MAIN RESULT) ──────────────────────────────────────
echo "=== Step 3: Searching with IVRG index ==="
"$BUILD_DIR/search_ivrg" \
    --index   "$IVRG_INDEX" \
    --data    "$BASE_FBIN"  \
    --queries "$QUERY_FBIN" \
    --gt      "$GT_IBIN"    \
    --K 10 --L "$L_LIST"
echo ""

# ─── 4. Vamana baseline ───────────────────────────────────────────────────────
echo "=== Step 4: Vamana baseline ==="
if [ ! -f "$VAMANA_INDEX" ]; then
    echo "  Building Vamana index..."
    "$BUILD_DIR/build_index" \
        --data "$BASE_FBIN" --output "$VAMANA_INDEX" \
        --R 32 --L 75 --alpha 1.2 --gamma 1.5
fi
"$BUILD_DIR/search_index" \
    --index   "$VAMANA_INDEX" \
    --data    "$BASE_FBIN"    \
    --queries "$QUERY_FBIN"   \
    --gt      "$GT_IBIN"      \
    --K 10 --L "$L_LIST"
echo ""

# ─── 5. nprobe sweep  (key ablation) ─────────────────────────────────────────
echo "=== Step 5: nprobe sweep ==="
echo "  Shows how many routing seeds are needed."
echo "  nprobe=1 = single best cell.  nprobe=3 = default."
for NP in 1 2 3 5; do
    IDX="$DATA_DIR/sift_ivrg_np${NP}.bin"
    echo ""
    echo "--- nprobe = ${NP} ---"
    if [ ! -f "$IDX" ]; then
        "$BUILD_DIR/build_ivrg" \
            --data "$BASE_FBIN" --output "$IDX" \
            --R 32 --L 75 --alpha 1.2 --gamma 1.5 \
            --K 512 --nprobe "$NP" --T 15
    fi
    "$BUILD_DIR/search_ivrg" \
        --index "$IDX" --data "$BASE_FBIN" \
        --queries "$QUERY_FBIN" --gt "$GT_IBIN" \
        --K 10 --L 10,20,50,100,200
done
echo ""

echo "=== Done! ==="
echo ""
echo "To plot: python3 scripts/plot_pareto.py tmp/results_ivrg.txt"