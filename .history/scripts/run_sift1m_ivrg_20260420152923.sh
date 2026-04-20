#!/usr/bin/env bash
#
# IVRG benchmark on SIFT1M — mirrors run_sift1m.sh exactly.
# Usage: ./scripts/run_sift1m_ivrg.sh
#
# Prerequisites: run ./scripts/run_sift1m.sh once so that
#   tmp/sift_base.fbin, tmp/sift_query.fbin, tmp/sift_gt.ibin exist.
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
echo "=== Step 1: Building project (including IVRG targets) ==="
mkdir -p "$BUILD_DIR"
pushd "$BUILD_DIR" > /dev/null
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
popd > /dev/null
echo ""

# ─── 2. Confirm data ──────────────────────────────────────────────────────────
echo "=== Step 2: Using existing SIFT1M binary files ==="
echo "  $BASE_FBIN"
echo "  $QUERY_FBIN"
echo "  $GT_IBIN"
echo ""

# ─── 3. Build IVRG index (K=512, nprobe=3) ────────────────────────────────────
echo "=== Step 3: Building IVRG index ==="
echo "  Vamana: R=32 L=75 alpha=1.2 gamma=1.5"
echo "  Routing: K=512 nprobe=3 T=15"
"$BUILD_DIR/build_ivrg" \
    --data   "$BASE_FBIN" \
    --output "$IVRG_INDEX" \
    --R 32 --L 75 --alpha 1.2 --gamma 1.5 \
    --K 512 --nprobe 3 --T 15
echo ""

# ─── 4. Search with IVRG (main result) ────────────────────────────────────────
echo "=== Step 4: Searching with IVRG index ==="
"$BUILD_DIR/search_ivrg" \
    --index   "$IVRG_INDEX" \
    --data    "$BASE_FBIN"  \
    --queries "$QUERY_FBIN" \
    --gt      "$GT_IBIN"    \
    --K 10 --L "$L_LIST"
echo ""

# ─── 5. Baseline: Vamana ──────────────────────────────────────────────────────
echo "=== Step 5: Searching with Vamana index (baseline) ==="
if [ ! -f "$VAMANA_INDEX" ]; then
    echo "  Vamana index not found — building now..."
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

# ─── 6. nprobe sweep (key ablation — cost of routing vs. recall gain) ─────────
echo "=== Step 6: nprobe sweep (K=512, vary nprobe) ==="
for NP in 1 2 3 5 8; do
    IDX="$DATA_DIR/sift_ivrg_np${NP}.bin"
    echo ""
    echo "--- nprobe = ${NP} ---"
    # Only rebuild if the index doesn't exist
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

# ─── 7. K sweep (number of Voronoi cells) ─────────────────────────────────────
echo "=== Step 7: K-clusters sweep (nprobe=3, vary K) ==="
for K in 128 256 512 1024; do
    IDX="$DATA_DIR/sift_ivrg_K${K}.bin"
    echo ""
    echo "--- K = ${K} ---"
    if [ ! -f "$IDX" ]; then
        "$BUILD_DIR/build_ivrg" \
            --data "$BASE_FBIN" --output "$IDX" \
            --R 32 --L 75 --alpha 1.2 --gamma 1.5 \
            --K "$K" --nprobe 3 --T 15
    fi
    "$BUILD_DIR/search_ivrg" \
        --index "$IDX" --data "$BASE_FBIN" \
        --queries "$QUERY_FBIN" --gt "$GT_IBIN" \
        --K 10 --L 10,20,50,100,200
done
echo ""

echo "=== Done! ==="
echo ""
echo "Capture: ./scripts/run_sift1m_ivrg.sh | tee tmp/results_ivrg.txt"
echo "Compare: diff <(grep -A8 'Step 4') <(grep -A8 'Step 5') tmp/results_ivrg.txt"
