#!/usr/bin/env bash
# =============================================================================
# run_sift1m_jl.sh  — Build and run the Vamana-JL (JL projection) variant
# Usage: ./scripts/run_sift1m_jl.sh [proj_dim]
#   proj_dim: target dimension after JL projection (default: 32)
#             Set to 0 to disable projection and run as standard Vamana.
# =============================================================================

set -e

PROJ_DIM=${1:-100}          # default: 128D -> 32D (4x speedup on distances)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
BUILD_DIR="$ROOT/build"
DATA_DIR="$ROOT/data/sift1m"

BASE="$ROOT/tmp/sift_base.fbin"
QUERY="$ROOT/tmp/sift_query.fbin"
GT="$ROOT/tmp/sift_gt.ibin"
INDEX="$ROOT/tmp/sift1m_jl.bin"

# ---- Parameters ----
R=32
L_BUILD=200
ALPHA=1.2
GAMMA=1.5
K=10
L_SEARCH="10,20,50,75,100,150,200"

# =============================================================================
echo "=== Step 1: Building the project ==="
# =============================================================================
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD_DIR" --target build_index_jl search_index_jl -j$(nproc)

# =============================================================================
echo ""
echo "=== Step 2: Converting data (if needed) ==="
# =============================================================================
if [ ! -f "$BASE" ]; then
    echo "  Converting sift_base.fvecs -> sift_base.fbin"
    python3 "$SCRIPT_DIR/convert_vecs.py" \
        "$DATA_DIR/sift_base.fvecs" "$BASE"
else
    echo "  $BASE already exists, skipping."
fi

if [ ! -f "$QUERY" ]; then
    echo "  Converting sift_query.fvecs -> sift_query.fbin"
    python3 "$SCRIPT_DIR/convert_vecs.py" \
        "$DATA_DIR/sift_query.fvecs" "$QUERY"
else
    echo "  $QUERY already exists, skipping."
fi

if [ ! -f "$GT" ]; then
    echo "  Converting sift_groundtruth.ivecs -> sift_groundtruth.ibin"
    python3 "$SCRIPT_DIR/convert_vecs.py" \
        "$DATA_DIR/sift_groundtruth.ivecs" "$GT"
else
    echo "  $GT already exists, skipping."
fi

# =============================================================================
echo ""
echo "=== Step 3: Building Vamana-JL index (proj_dim=${PROJ_DIM}) ==="
# =============================================================================
"$BUILD_DIR/build_index_jl" \
    --data     "$BASE"    \
    --output   "$INDEX"   \
    --R        $R         \
    --L_build  $L_BUILD   \
    --alpha    $ALPHA     \
    --gamma    $GAMMA     \
    --proj_dim $PROJ_DIM

# =============================================================================
echo ""
echo "=== Step 4: Searching Vamana-JL index ==="
# =============================================================================
"$BUILD_DIR/search_index_jl" \
    --index   "$INDEX"    \
    --data    "$BASE"     \
    --queries "$QUERY"    \
    --gt      "$GT"       \
    --K       $K          \
    --L       $L_SEARCH

echo ""
echo "=== Done ==="