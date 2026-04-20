# GraphANN — Educational Vamana Index

A clean, modular C++ implementation of the **Vamana graph-based approximate nearest neighbor (ANN) index** for students to learn, experiment with, and extend.

Implements the core algorithm from [*DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html) (NeurIPS 2019), along with several algorithmic improvements explored in our Milestone 2 report.

---

## Algorithm Overview

### Build Phase
For each point (in a random order, parallelized with OpenMP):

1. **Greedy Search**: Search the current graph for the point itself, producing a candidate list of size `L`
2. **Robust Prune (α-RNG)**: Prune candidates to at most `R` diverse neighbors using the alpha-RNG rule — a candidate `c` is kept only if `dist(node, c) ≤ α · dist(c, n)` for all already-selected neighbors `n`
3. **Add Edges**: Set forward edges; add backward edges to each neighbor
4. **Degree Check**: If any neighbor's degree exceeds `γR`, prune its neighborhood

Per-node mutexes ensure correctness during parallel construction.

### Search Phase
Greedy beam search starting from a fixed start node, maintaining a candidate set bounded at size `L`. Returns the top-`K` closest points found.

### Parameters
| Parameter | Typical Range | Description |
|-----------|--------------|-------------|
| `R` | 32–64 | Max out-degree (graph connectivity) |
| `L` (build) | 75–200 | Search list size during construction (≥ R) |
| `α` (alpha) | 1.0–1.5 | RNG pruning threshold (> 1 keeps long-range edges) |
| `γ` (gamma) | 1.2–1.5 | Degree multiplier triggering neighbor pruning |
| `L` (search) | 10–200 | Search list size at query time (≥ K) |
| `K` | 1–100 | Number of nearest neighbors to return |

---

## Implemented Improvements

### 1. Medoid Start Node
Replaces the random start node with the **medoid** — the dataset point closest to the centroid. This minimises expected initial distance to any query, reducing approach hops before useful exploration.

- **Result**: Avg. Dist. Cmps drops 642.4 → 537.1 (−16%) at L=10; Recall@10 improves 0.776 → 0.797.

### 2. Two-Pass Graph Construction
Runs Vamana construction twice:
- **Pass 1** (α=1.0): Strict monotone RNG, well-connected base graph.
- **Pass 2** (α=1.2): Relaxed RNG over a re-shuffled order, adding long-range diversity edges.

Early-inserted nodes get re-evaluated against the fully-populated graph in pass 2.

- **Result**: Combined with medoid start, L=10 latency drops 187 → 172 µs and L=100 P99 latency drops 6641 → 3984 µs (−40%).

### 3. Catastrophic Forgetting Mitigation
After construction, nodes with out-degree below δ_min = ⌊R/2⌋ are identified and re-pruned against a fresh beam search candidate list from the fully-populated graph. This fixes suboptimal neighbor sets for early-inserted nodes.

- **Result**: Fraction of under-connected nodes drops from 8.3% to 1.1%; further −3% in Avg. Dist. Cmps at L=10.

### 4. Radial Layer Graph (RLG)
Augments the standard α-RNG pruning with explicit **geometric shell coverage**. The neighbor budget R is split:
- **Stage 1** (¾R slots): Standard α-RNG neighbors.
- **Stage 2** (¼R slots): Closest candidate from each unrepresented geometric shell around the node.

Shells are defined as `[m^(k-1)·r_p, m^k·r_p)` for multiplier m (default m=2), motivated by Kleinberg's theorem on navigable small-world graphs.

- **Result**: Modest improvement over the Vamana baseline, especially at mid-recall operating points (L=20–50).

### 5. Inverted Voronoi Routing Graph (IVRG) ⭐ Best Result
Replaces the fixed medoid start with **query-adaptive Voronoi seeding**. At build time, k-means++ clustering partitions the dataset into K Voronoi cells. At query time, the nearest `nprobe` cluster representatives are found in O(Kd) operations and used as beam search seeds alongside the global medoid.

The graph structure is **identical to Vamana** — only the seed set changes. Because the medoid is always included, IVRG is never worse than Vamana at any fixed L.

- **Result**: −14% Avg. Dist. Cmps at L=50 (recall ≈ 0.97) and −26% at L=100 vs. the Vamana baseline.

### 6. Vamana-JL (Johnson–Lindenstrauss Projection)
Reduces build-time distance cost by projecting all vectors to k=100 dimensions (from 128) using a random Gaussian matrix before construction. Search also operates in projected space.

- **Result**: Lower recall ceiling on SIFT1M (effective intrinsic dim ≈ 10 makes rank ordering sensitive to projection). Viable for genuinely high-dimensional data (d > 512).

---

## Project Structure

```
graphann/
├── include/                        # Headers for all implementations
│   ├── distance.h                  # Squared L2 distance
│   ├── io_utils.h                  # fbin/ibin file loaders
│   ├── ivrg_index.h                
│   ├── jl_projection.h             # JL random projection
│   ├── rlg_index.h                 
│   ├── timer.h                     # chrono-based timer
│   ├── vamana_index_jl.h           
│   └── vamana_index.h 
|            
├── results/                        # Benchmark outputs
│   ├── compare_recall_distcmps.png 
│   ├── compare_recall_latency.png 
│   └── compare_results.csv     
|    
├── scripts/                        # Automation and plotting
│   ├── convert_vecs.py             # fvecs/ivecs → fbin/ibin
│   ├── plot_pareto.py              
│   ├── plots.py                    
│   ├── run_sift1m_ivrg.sh          # End-to-end IVRG on SIFT1M
│   ├── run_sift1m_jl.sh            # End-to-end Vamana-JL on SIFT1M
│   ├── run_sift1m_rlg.sh           # End-to-end RLG on SIFT1M
│   └── run_sift1m.sh               # End-to-end baseline Vamana on SIFT1M
|
├── src/                            # Baseline Vamana + all build improvements
│   ├── build_index.cpp             
│   ├── distance.cpp                
│   ├── io_utils.cpp               
│   ├── search_index.cpp          
│   └── vamana_index.cpp          
|  
├── src_ivrg/                       # IVRG: Inverted Voronoi Root Graph
│   ├── build_ivrg.cpp              
│   ├── ivrg_index.cpp              
│   └── search_ivrg.cpp    
|         
├── src_jl/                         
│   ├── build_index_jl.cpp          
│   ├── search_index_jl.cpp         
│   └── vamana_index_jl.cpp  
|       
├── src2/                           # RLG: Radial Layer Graph
│   ├── build_rlg.cpp               
│   ├── rlg_index.cpp            
│   └── search_rlg.cpp   
|         
├── .gitignore
├── CMakeLists.txt                  # C++17, OpenMP, -O3 -march=native
├── README.md
└── Report.pdf                      # Milestone 2 report
```

### Key files to study
- **`src/vamana_index.cpp`** — the core algorithm: `greedy_search()`, `robust_prune()`, `build()`
- **`include/vamana_index.h`** — data structures (adjacency list graph, per-node locks)

---

## Build

Requirements: C++17 compiler with OpenMP support (GCC ≥ 7, Clang ≥ 10).

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

This produces two executables: `build_index` and `search_index`.

---

## Quick Start — SIFT1M end-to-end

A single script downloads the [SIFT1M](http://corpus-texmex.irisa.fr/) dataset, converts it to binary format, builds a Vamana index, and runs search with recall evaluation:

```bash
./scripts/run_sift1m.sh
```

This will:
1. Build the project (cmake + make)
2. Download SIFT1M (1M base vectors, 10K queries, ground truth) into `tmp/sift/`
3. Convert `.fvecs`/`.ivecs` files to `.fbin`/`.ibin` format in `tmp/`
4. Build a Vamana index with default parameters (R=32, L=75, α=1.2, γ=1.5)
5. Run search at multiple `L` values and report recall@10, latency, and distance computations

Requires: `curl`, `python3` with `numpy`, and a C++17 compiler with OpenMP.

---

## Usage

### File Formats

**fbin** (float binary): Used for dataset and query vectors.
```
[4 bytes: uint32 npts] [4 bytes: uint32 dims] [npts * dims * 4 bytes: float32 row-major vectors]
```

**ibin** (int binary): Used for ground truth neighbor IDs.
```
[4 bytes: uint32 npts] [4 bytes: uint32 dims] [npts * dims * 4 bytes: uint32 row-major IDs]
```

Standard ANN benchmark datasets (SIFT, GIST, GloVe, etc.) are available in this format from [ANN Benchmarks](http://ann-benchmarks.com/) and [big-ann-benchmarks](https://big-ann-benchmarks.com/).

---

## Build and Search Commands — Per Method

All methods share the same base build parameters: `R=32`, `L=75`, `α=1.2`, `γ=1.5`.

---

### Baseline Vamana (random start, single-pass)

```bash
./build_index \
  --data /path/to/base.fbin \
  --output /path/to/index_baseline.bin \
  --R 32 --L 75 --alpha 1.2 --gamma 1.5

./search_index \
  --index /path/to/index_baseline.bin \
  --data /path/to/base.fbin \
  --queries /path/to/query.fbin \
  --gt /path/to/gt.ibin \
  --K 10 \
  --L 10,20,30,50,75,100,150,200
```

---

### Vamana + Medoid Start Node

```bash
./build_index \
  --data /path/to/base.fbin \
  --output /path/to/index_medoid.bin \
  --R 32 --L 75 --alpha 1.2 --gamma 1.5 \
  --medoid

./search_index \
  --index /path/to/index_medoid.bin \
  --data /path/to/base.fbin \
  --queries /path/to/query.fbin \
  --gt /path/to/gt.ibin \
  --K 10 \
  --L 10,20,30,50,75,100,150,200
```

---

### Vamana + Medoid Start + Two-Pass Construction

```bash
./build_index \
  --data /path/to/base.fbin \
  --output /path/to/index_twopass.bin \
  --R 32 --L 75 --alpha 1.2 --gamma 1.5 \
  --medoid --two-pass --alpha2 1.2

./search_index \
  --index /path/to/index_twopass.bin \
  --data /path/to/base.fbin \
  --queries /path/to/query.fbin \
  --gt /path/to/gt.ibin \
  --K 10 \
  --L 10,20,30,50,75,100,150,200
```

---

### Vamana + Medoid + Two-Pass + Catastrophic Forgetting Mitigation

```bash
./build_index \
  --data /path/to/base.fbin \
  --output /path/to/index_cf.bin \
  --R 32 --L 75 --alpha 1.2 --gamma 1.5 \
  --medoid --two-pass --alpha2 1.2 \
  --fix-catastrophic --delta-min 16

./search_index \
  --index /path/to/index_cf.bin \
  --data /path/to/base.fbin \
  --queries /path/to/query.fbin \
  --gt /path/to/gt.ibin \
  --K 10 \
  --L 10,20,30,50,75,100,150,200
```

---

### Radial Layer Graph (RLG)

```bash
./build_index \
  --data /path/to/base.fbin \
  --output /path/to/index_rlg.bin \
  --R 32 --L 75 --alpha 1.2 --gamma 1.5 \
  --medoid --two-pass --alpha2 1.2 \
  --rlg --shell-multiplier 2.0 --shell-slots 8

./search_index \
  --index /path/to/index_rlg.bin \
  --data /path/to/base.fbin \
  --queries /path/to/query.fbin \
  --gt /path/to/gt.ibin \
  --K 10 \
  --L 10,20,30,50,75,100,150,200
```

---

### Inverted Voronoi Routing Graph (IVRG)

```bash
# Build: identical two-pass Vamana graph + cluster the dataset
./build_index \
  --data /path/to/base.fbin \
  --output /path/to/index_ivrg.bin \
  --R 32 --L 75 --alpha 1.2 --gamma 1.5 \
  --medoid --two-pass --alpha2 1.2 \
  --ivrg --clusters 512 --lloyd-iters 15

# Search: seed beam search from nearest Voronoi representatives
./search_index \
  --index /path/to/index_ivrg.bin \
  --data /path/to/base.fbin \
  --queries /path/to/query.fbin \
  --gt /path/to/gt.ibin \
  --K 10 \
  --L 10,20,30,50,75,100,150,200 \
  --ivrg --nprobe 3
```

---

### Vamana-JL (Johnson–Lindenstrauss Projection)

```bash
# Build: project to k dimensions, then run two-pass Vamana on projected vectors
./build_index \
  --data /path/to/base.fbin \
  --output /path/to/index_jl.bin \
  --R 32 --L 75 --alpha 1.2 --gamma 1.5 \
  --medoid --two-pass --alpha2 1.2 \
  --jl --jl-dims 100

# Search: query is projected before beam search
./search_index \
  --index /path/to/index_jl.bin \
  --data /path/to/base.fbin \
  --queries /path/to/query.fbin \
  --gt /path/to/gt.ibin \
  --K 10 \
  --L 10,20,30,50,75,100,150,200 \
  --jl --jl-dims 100
```

---

## Output Format

```
=== Search Results (K=10) ===
       L     Recall@10   Avg Dist Cmps  Avg Latency (us)  P99 Latency (us)
--------------------------------------------------------------------------
      10         0.7950           416.7             140.5             389.2
      50         0.9680          1135.0             380.7             921.4
     100         0.9898          1607.3             632.5            1843.7
     200         0.9968          2835.9            1932.1            5102.3
```

---

## Performance Notes

- **Parallelism**: OpenMP `parallel for schedule(dynamic)` for both build (point insertion) and search (queries)
- **Memory layout**: Contiguous row-major float arrays, 64-byte aligned for SIMD
- **Vectorization**: `-O3 -march=native` auto-vectorizes the L2 distance loop — no manual intrinsics needed
- **Lock granularity**: Per-node `std::mutex` — threads only contend when updating the *same* node's adjacency list
- **Concurrent search**: Queries are dispatched across OpenMP threads with dynamic scheduling (chunk size 1); prefetch-aware neighbor expansion hides memory latency during graph traversal
- **No external dependencies** beyond OpenMP

---

## Results Summary (SIFT1M, K=10)

| Algorithm | L | Recall@10 | Avg Cmps | Avg Lat (µs) |
|-----------|---|-----------|----------|--------------|
| Baseline (random start) | 10 | 0.776 | 642.4 | 215.1 |
| + Medoid start | 10 | 0.797 | 537.1 (−16%) | 186.9 |
| + Two-pass build | 10 | 0.777 | 477.7 (−26%) | 171.6 |
| **IVRG** | 10 | **0.7950** | **416.7** | **140.5** |
| **IVRG** | 100 | **0.9898** | **1607.3** | **632.5** |
| RLG | 10 | 0.7970 | 497.9 | 147.6 |
| Vamana-JL (k=100) | 10 | 0.6310 | 515.3 | 148.8 |

---

## Some sample things to try, and start experimenting with!

0. **Code understanding**: Use AI tools to understand the logic of the algorithm and how it is a heuristic approximation of what we discussed in class

1. **Beam width experiments**: Try different `L` values during build and measure recall vs build time. What's the sweet spot?

2. **Medoid start node**: Replace the random start node with the *medoid* — the point closest to the centroid of the dataset. How does this affect search recall?

3. **Change the edges in index build**: Run the build twice — second pass starts from the graph produced by the first. How does recall change?

4. **Change the search algorithm**: Plot the histogram of node degrees. Is it uniform? What happens with different `α` values?

5. **Concurrent search optimization**: Replace `std::vector<bool> visited` in `greedy_search()` with a pre-allocated scratch buffer to avoid per-query allocation.

6. **IVRG nprobe sweep**: Vary `--nprobe` from 1 to 8 and plot recall vs. distance computations. What is the sweet spot?

7. **IVRG + RLG combined**: Stack the IVRG routing layer on an RLG graph — both improvements are orthogonal and should compose.

8. **Vamana-JL on high-d data**: Generate a synthetic dataset at d=512 or d=1024 where the projection speedup is more pronounced.

---

## References

- Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*, NeurIPS 2019
- Malkov & Yashunin, *Efficient and Robust ANN Search Using Hierarchical Navigable Small World Graphs*, IEEE TPAMI 2020
- Kleinberg, *Navigation in a Small World*, Nature 2000
- Johnson & Lindenstrauss, *Extensions of Lipschitz Mappings into a Hilbert Space*, 1984
- Johnson, Douze & Jégou, *Billion-Scale Similarity Search with GPUs*, IEEE Trans. Big Data 2021