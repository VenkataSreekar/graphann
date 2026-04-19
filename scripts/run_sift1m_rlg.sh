#!/usr/bin/env bash
#
# Builds the Radial Layer Graph index on SIFT1M and benchmarks it
# against the Vamana index, matching the structure of run_sift1m.sh exactly.
#
# Usage: ./scripts/run_sift1m_rlg.sh
#
# Prerequisites:
#   1. Run ./scripts/run_sift1m.sh at least once so that:
#        - SIFT1M is already downloaded into tmp/sift/
#        - tmp/sift_base.fbin, tmp/sift_query.fbin, tmp/sift_gt.ibin exist
#        - The existing build_index / search_index binaries are already built
#   2. You have appended CMakeLists_patch.txt to your CMakeLists.txt
#   3. New files are in place:
#        include/rlg_index.h
#        src/rlg_index.cpp
#        src/build_rlg.cpp
#        src/search_rlg.cpp
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/tmp"
BUILD_DIR="$ROOT/build"

# Reuse the exact same converted files that run_sift1m.sh produced
BASE_FBIN="$DATA_DIR/sift_base.fbin"
QUERY_FBIN="$DATA_DIR/sift_query.fbin"
GT_IBIN="$DATA_DIR/sift_gt.ibin"

# RLG index output (separate file from the Vamana index)
RLG_INDEX="$DATA_DIR/sift_rlg_index.bin"

# ─── 0. Sanity-check that the converted data files exist ─────────────────────
echo "=== Pre-flight checks ==="
for f in "$BASE_FBIN" "$QUERY_FBIN" "$GT_IBIN"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        echo "       Please run ./scripts/run_sift1m.sh first to download"
        echo "       and convert SIFT1M data."
        exit 1
    fi
done
echo "Data files OK."
echo ""

# ─── 1. Build the project (compiles build_rlg + search_rlg alongside existing targets)
echo "=== Step 1: Building the project (including RLG targets) ==="
mkdir -p "$BUILD_DIR"
pushd "$BUILD_DIR" > /dev/null
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
popd > /dev/null
echo ""

# ─── 2. Data already downloaded and converted by run_sift1m.sh ───────────────
echo "=== Step 2: Using existing SIFT1M binary files ==="
echo "  $BASE_FBIN"
echo "  $QUERY_FBIN"
echo "  $GT_IBIN"
echo ""

# ─── 3. Build the RLG index ──────────────────────────────────────────────────
echo "=== Step 3: Building the Radial Layer Graph index ==="
echo "  R=32  L=75  m=2.0  alpha=0.5  two_pass=1"
"$BUILD_DIR/build_rlg" \
    --data      "$BASE_FBIN"  \
    --output    "$RLG_INDEX"  \
    --R    32                 \
    --L    75                 \
    --m    2.0                \
    --alpha 0.5               \
    --two_pass 1
echo ""

# ─── 4. Search with the RLG index ────────────────────────────────────────────
echo "=== Step 4: Searching with RLG index ==="
"$BUILD_DIR/search_rlg" \
    --index   "$RLG_INDEX"   \
    --data    "$BASE_FBIN"   \
    --queries "$QUERY_FBIN"  \
    --gt      "$GT_IBIN"     \
    --K  10                  \
    --L  10,20,30,50,75,100,150,200
echo ""

# ─── 5. Run baseline Vamana search for comparison ────────────────────────────
echo "=== Step 5: Searching with Vamana index (baseline) ==="
VAMANA_INDEX="$DATA_DIR/sift_index.bin"
if [ ! -f "$VAMANA_INDEX" ]; then
    echo "Vamana index not found at $VAMANA_INDEX"
    echo "Building it now with default parameters ..."
    "$BUILD_DIR/build_index" \
        --data   "$BASE_FBIN"   \
        --output "$VAMANA_INDEX" \
        --R 32 --L 75 --alpha 1.2 --gamma 1.5
fi

"$BUILD_DIR/search_index" \
    --index   "$VAMANA_INDEX" \
    --data    "$BASE_FBIN"    \
    --queries "$QUERY_FBIN"   \
    --gt      "$GT_IBIN"      \
    --K  10                   \
    --L  10,20,30,50,75,100,150,200
echo ""

# ─── 6. Parameter sweep: different m values ──────────────────────────────────
echo "=== Step 6: RLG m-value sweep (m = 1.5, 2.0, 2.5, 3.0) ==="
for M in 1.5 2.0 2.5 3.0; do
    RLG_M_INDEX="$DATA_DIR/sift_rlg_m${M}.bin"
    echo ""
    echo "--- m = ${M} ---"

    "$BUILD_DIR/build_rlg" \
        --data   "$BASE_FBIN"    \
        --output "$RLG_M_INDEX"  \
        --R 32 --L 75 --m "$M" --alpha 0.5 --two_pass 1

    "$BUILD_DIR/search_rlg" \
        --index   "$RLG_M_INDEX" \
        --data    "$BASE_FBIN"   \
        --queries "$QUERY_FBIN"  \
        --gt      "$GT_IBIN"     \
        --K 10                   \
        --L 10,20,50,100,200
done
echo ""

# ─── 7. Alpha sweep: angular diversity threshold ──────────────────────────────
echo "=== Step 7: RLG alpha sweep (alpha = 0.0, 0.3, 0.5, 0.7, 0.9) ==="
for A in 0.0 0.3 0.5 0.7 0.9; do
    RLG_A_INDEX="$DATA_DIR/sift_rlg_a${A}.bin"
    echo ""
    echo "--- alpha = ${A} ---"

    "$BUILD_DIR/build_rlg" \
        --data   "$BASE_FBIN"    \
        --output "$RLG_A_INDEX"  \
        --R 32 --L 75 --m 2.0 --alpha "$A" --two_pass 1

    "$BUILD_DIR/search_rlg" \
        --index   "$RLG_A_INDEX" \
        --data    "$BASE_FBIN"   \
        --queries "$QUERY_FBIN"  \
        --gt      "$GT_IBIN"     \
        --K 10                   \
        --L 10,20,50,100,200
done
echo ""

echo "=== Done! ==="
echo ""
echo "Results summary:"
echo "  Step 4  = RLG Pareto curve    (main result)"
echo "  Step 5  = Vamana Pareto curve (baseline to compare against)"
echo "  Step 6  = Effect of m on recall/QPS"
echo "  Step 7  = Effect of angular diversity on recall/QPS"
echo ""
echo "Pipe to a file with:  ./scripts/run_sift1m_rlg.sh | tee results_rlg.txt"
