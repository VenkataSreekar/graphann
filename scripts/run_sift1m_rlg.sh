#!/bin/bash
# run_sift1m_rlg.sh
# Downloads SIFT1M, builds both Vamana and RLG, runs comparison.
#
# Run from the radial_layer_graph/ directory:
#   chmod +x scripts/run_sift1m_rlg.sh && ./scripts/run_sift1m_rlg.sh

set -e

DATA_DIR="data/sift1m"
BUILD_DIR="build"

echo "=== Radial Layer Graph — SIFT1M Benchmark ==="

# ---- Download SIFT1M if needed ----
mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/sift_base.fvecs" ]; then
    echo "[1/5] Downloading SIFT1M..."
    wget -q --show-progress -P "$DATA_DIR" \
        http://corpus-texmex.irisa.fr/ftp/base/ANN_SIFT1M.tar.gz
    tar -xzf "$DATA_DIR/ANN_SIFT1M.tar.gz" -C "$DATA_DIR" --strip-components=1
    echo "Downloaded SIFT1M."
else
    echo "[1/5] SIFT1M already downloaded."
fi

# ---- Convert .fvecs/.ivecs → .fbin/.ibin ----
if [ ! -f "$DATA_DIR/sift_base.fbin" ]; then
    echo "[2/5] Converting to .fbin/.ibin..."
    python3 scripts/convert_vecs.py "$DATA_DIR/sift_base.fvecs"    "$DATA_DIR/sift_base.fbin"
    python3 scripts/convert_vecs.py "$DATA_DIR/sift_query.fvecs"   "$DATA_DIR/sift_query.fbin"
    python3 scripts/convert_vecs.py "$DATA_DIR/sift_groundtruth.ivecs" "$DATA_DIR/sift_groundtruth.ibin" --int
    echo "Converted."
else
    echo "[2/5] Already converted."
fi

# ---- Build ----
echo "[3/5] Building C++ binaries..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" > /dev/null
make -j$(nproc) 2>&1 | tail -5
cd ..
echo "Build complete."

# ---- Head-to-head comparison ----
echo "[4/5] Running head-to-head comparison (R=32, m=2.0, L_build=100)..."
"$BUILD_DIR/compare" \
    "$DATA_DIR/sift_base.fbin" \
    "$DATA_DIR/sift_query.fbin" \
    "$DATA_DIR/sift_groundtruth.ibin" \
    32 2.0 100 10 \
    | tee results_comparison.txt

# ---- Parameter sweep: different m values ----
echo ""
echo "[5/5] Sweeping m values (m=1.5, 2.0, 2.5, 3.0)..."
for M in 1.5 2.0 2.5 3.0; do
    echo ""
    echo "--- m = $M ---"
    "$BUILD_DIR/build_rlg" \
        "$DATA_DIR/sift_base.fbin" \
        "rlg_m${M}.bin" \
        32 $M 100 0.5 1
    "$BUILD_DIR/search_rlg" \
        "rlg_m${M}.bin" \
        "$DATA_DIR/sift_base.fbin" \
        "$DATA_DIR/sift_query.fbin" \
        "$DATA_DIR/sift_groundtruth.ibin" \
        10 100
done | tee results_m_sweep.txt

echo ""
echo "=== Done ==="
echo "Results saved to: results_comparison.txt  results_m_sweep.txt"
echo "Run: python3 scripts/plot_results.py  to generate Pareto plots"
