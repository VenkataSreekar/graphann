#!/usr/bin/env bash
#
# Downloads SIFT data, builds Vamana vs RLG comparison.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/tmp"
BUILD_DIR="$ROOT/build"
SIFT_DIR="$DATA_DIR/sift"

# Using the URL that we know works for you
SIFT_URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT_TAR="$DATA_DIR/sift.tar.gz"

# Output binary files
BASE_FBIN="$DATA_DIR/sift_base.fbin"
QUERY_FBIN="$DATA_DIR/sift_query.fbin"
GT_IBIN="$DATA_DIR/sift_gt.ibin"

echo "=== Radial Layer Graph — SIFT Benchmark ==="

# ─── 1. Download SIFT (Using your working logic) ─────────────────────────────
echo "[1/5] Checking SIFT data..."
mkdir -p "$DATA_DIR"
if [ -d "$SIFT_DIR" ] && [ -f "$SIFT_DIR/sift_base.fvecs" ]; then
    echo "SIFT data already exists, skipping download."
else
    echo "Downloading from $SIFT_URL ..."
    curl -L -o "$SIFT_TAR" "$SIFT_URL"
    echo "Extracting..."
    tar -xzf "$SIFT_TAR" -C "$DATA_DIR"
    rm -f "$SIFT_TAR"
fi

# ─── 2. Convert to fbin / ibin ───────────────────────────────────────────────
echo "[2/5] Converting to .fbin/.ibin..."
if [ -f "$BASE_FBIN" ] && [ -f "$QUERY_FBIN" ] && [ -f "$GT_IBIN" ]; then
    echo "Binary files already exist, skipping conversion."
else
    python3 "$ROOT/scripts/convert_vecs.py" "$SIFT_DIR/sift_base.fvecs" "$BASE_FBIN"
    python3 "$ROOT/scripts/convert_vecs.py" "$SIFT_DIR/sift_query.fvecs" "$QUERY_FBIN"
    # Note: RLG script expects --int flag or similar for ground truth
    python3 "$ROOT/scripts/convert_vecs.py" "$SIFT_DIR/sift_groundtruth.ivecs" "$GT_IBIN"
fi

# ─── 3. Build the project ────────────────────────────────────────────────────
echo "[3/5] Building C++ binaries..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
cd ..

# ─── 4. Head-to-head comparison ──────────────────────────────────────────────
echo "[4/5] Running head-to-head comparison..."
# This runs the 'compare' binary created by your CMakeLists.txt
"$BUILD_DIR/compare" \
    "$BASE_FBIN" \
    "$QUERY_FBIN" \
    "$GT_IBIN" \
    32 2.0 100 10 | tee results_comparison.txt

# ─── 5. Generate Graph ───────────────────────────────────────────────────────
echo "[5/5] Generating Pareto curve..."
if [ -f "$ROOT/scripts/plot_results.py" ]; then
    python3 "$ROOT/scripts/plot_results.py"
fi

echo "=== Done! Results are in the 'results' folder ==="