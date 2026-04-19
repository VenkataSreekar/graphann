#pragma once
// rlg_index.h  — Radial Layer Graph ANN index
// Place in graphann/include/ alongside vamana_index.h
//
// ── Algorithm (Augmented Robust Prune) ──────────────────────────────────────
//
//  For every node p, we run TWO pruning stages:
//
//  Stage 1 — alpha-RNG (identical to Vamana's robust_prune):
//    Greedily keep candidate c if, for all already-selected neighbors s:
//      d(p,c)  <=  alpha * d(c,s)
//    This ensures navigation quality; output is never worse than Vamana.
//
//  Stage 2 — Shell Augmentation:
//    Using r_base = dist(p, nearest selected neighbor), define geometric shells:
//      Shell 0  :  [0,               r_base)
//      Shell k  :  [m^(k-1)*r_base,  m^k*r_base)   k = 1,2,...
//    For each shell NOT yet represented in Stage-1 output, add the closest
//    remaining candidate from that shell (if R budget allows).
//
//  Result: at least as good as Vamana, with explicit long-range edges
//  in shells that alpha-RNG did not naturally cover.
//
// ── Theoretical basis ───────────────────────────────────────────────────────
//  Kleinberg (STOC 2000): greedy routing achieves O(log² n) hops iff edge
//  lengths follow a power-law / geometric distribution.  Our shells produce
//  exactly this distribution while keeping Vamana's proven α-RNG base.
//
// ── Interface ────────────────────────────────────────────────────────────────
//  Intentionally mirrors VamanaIndex so search_rlg.cpp ≈ search_index.cpp.

#include "io_utils.h"   // FloatMatrix, IntMatrix, AlignedFree, load_fbin
#include "timer.h"      // Timer

#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// SearchResult is already declared in vamana_index.h.
// We redeclare it in the rlg namespace to avoid conflict when both headers
// are included.  search_rlg.cpp only includes rlg_index.h so it will use
// rlg::SearchResult.  If you include both headers in the same TU, just use
// the global one from vamana_index.h (they are identical).
namespace rlg {

struct SearchResult {
    std::vector<uint32_t> ids;
    uint32_t dist_cmps;
    double   latency_us;
};

class RLGIndex {
public:
    RLGIndex()  = default;
    ~RLGIndex() { if (owns_data_ && data_) { std::free(data_); data_ = nullptr; } }

    // ── Build ────────────────────────────────────────────────────────────────
    // Parameters match build_index.cpp convention + one new param (m):
    //   data_path : .fbin file  (same format as Vamana)
    //   R         : max out-degree per node
    //   L         : beam width during construction  (L >= R)
    //   alpha     : α-RNG diversity factor  (1.0–1.5, same as Vamana --alpha)
    //   m         : shell multiplier for augmentation  (try 1.5–3.0)
    //   gamma     : over-degree trigger = gamma*R  (same as Vamana --gamma)
    void build(const std::string& data_path,
               uint32_t R     = 32,
               uint32_t L     = 75,
               float    alpha = 1.2f,
               float    m     = 2.0f,
               float    gamma = 1.5f);

    // ── Search ───────────────────────────────────────────────────────────────
    // Identical signature to VamanaIndex::search so search_rlg.cpp
    // can be a near copy-paste of search_index.cpp.
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    // ── Persistence ──────────────────────────────────────────────────────────
    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    // ── Accessors ────────────────────────────────────────────────────────────
    uint32_t get_npts() const { return npts_; }
    uint32_t get_dim()  const { return dim_;  }

    // Degree distribution statistics (for benchmarking / reporting)
    void degree_stats(float& mean, float& stddev, float& gini) const;

private:
    using Candidate = std::pair<float, uint32_t>;   // (dist, id) — same as Vamana

    // ── Core algorithms ──────────────────────────────────────────────────────

    // Returns (sorted_results, dist_cmps)
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search(const float* query, uint32_t L) const;

    // Stage-1: alpha-RNG + Stage-2: shell augmentation
    void augmented_robust_prune(uint32_t node,
                                 std::vector<Candidate>& candidates,
                                 float alpha);

    // One full randomised pass over all points
    void run_build_pass(const std::vector<uint32_t>& perm,
                        uint32_t R, uint32_t L,
                        float alpha, uint32_t gamma_R);

    // Shell index: which geometric shell does distance d fall into?
    int dist_to_shell(float d, float r_base) const;

    // ── Data ─────────────────────────────────────────────────────────────────
    float*   data_      = nullptr;
    uint32_t npts_      = 0;
    uint32_t dim_       = 0;
    bool     owns_data_ = false;

    // ── Graph ────────────────────────────────────────────────────────────────
    std::vector<std::vector<uint32_t>> graph_;
    uint32_t start_node_ = 0;

    // ── RLG parameters ───────────────────────────────────────────────────────
    uint32_t R_    = 32;
    float    m_    = 2.0f;   // shell multiplier

    // ── Concurrency ──────────────────────────────────────────────────────────
    mutable std::vector<std::mutex> locks_;

    // ── Helpers ──────────────────────────────────────────────────────────────
    uint32_t compute_medoid() const;

    const float* get_vec(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }
};

} // namespace rlg