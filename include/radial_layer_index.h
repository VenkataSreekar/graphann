#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>
#include <limits>
#include <mutex>
#include <unordered_set>

// ============================================================
//  Radial Layer Graph (RLG) Index
//
//  Core idea (Kleinberg 2000 + multi-scale ANN):
//    For each node p, choose r_p = dist(p, nearest_neighbor).
//    Partition ALL possible neighbors into geometric shells:
//      Shell k : [m^(k-1)*r_p,  m^k*r_p)    k = 1,2,...,K
//    Select up to quota_k neighbors per shell, ensuring both
//    distance-scale diversity AND angular diversity within shell.
//
//  Parameters:
//    R       – total max degree per node  (like Vamana)
//    m       – geometric multiplier, ideally 2^(1/intrinsic_dim)
//    K       – number of shells (auto-computed to cover the data)
//    alpha   – angular diversity threshold within shell
//    L_build – candidate list size during construction
//    L_search– candidate list size during search
// ============================================================

namespace rlg {

// ---- distance helpers (L2 squared, SIMD-friendly layout) ----
inline float l2_sq(const float* a, const float* b, int dim) {
    float s = 0.f;
    for (int i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

inline float dot(const float* a, const float* b, int dim) {
    float s = 0.f;
    for (int i = 0; i < dim; ++i) s += a[i] * b[i];
    return s;
}

// cosine similarity between two DIFFERENCE vectors (for angular diversity)
inline float cos_sim(const float* va, const float* vb, int dim) {
    float ab = dot(va, vb, dim);
    float aa = dot(va, va, dim);
    float bb = dot(vb, vb, dim);
    if (aa < 1e-12f || bb < 1e-12f) return 0.f;
    return ab / std::sqrt(aa * bb);
}

struct Candidate {
    float dist;   // L2 squared distance to query/pivot
    int   id;
    bool operator<(const Candidate& o) const { return dist < o.dist; }
};

// ---- per-node shell assignment ----
struct ShellInfo {
    float r_base;          // = dist to nearest neighbor (innermost shell boundary)
    float m;               // geometric ratio
    int   K;               // number of shells used for this node
};

// ---- index structure ----
class RadialLayerIndex {
public:
    // ---- construction ----
    RadialLayerIndex() = default;

    // Main constructor
    // data     : row-major, n×dim
    // R        : max degree per node
    // m        : shell multiplier (try 1.5 to 2.5; default 2.0)
    // L_build  : beam width during build greedy search
    // alpha    : angular diversity factor (0=no angular filter, 1=strict)
    // two_pass : run a second refinement pass (improves early-insertion nodes)
    void build(const std::vector<float>& data, int n, int dim,
               int R       = 32,
               float m     = 2.0f,
               int L_build = 100,
               float alpha = 0.5f,
               bool two_pass = true);

    // Search: returns top-K neighbor IDs sorted by distance
    std::vector<int> search(const float* query, int K, int L_search = 100) const;

    // Batch search (parallel)
    std::vector<std::vector<int>> batch_search(
        const std::vector<float>& queries, int nq, int K, int L_search = 100) const;

    // ---- accessors for benchmarking ----
    const std::vector<std::vector<int>>& graph()       const { return adj_; }
    int n()     const { return n_; }
    int dim()   const { return dim_; }
    float m()   const { return m_; }
    int R()     const { return R_; }

    // Per-node shell info
    const ShellInfo& shell_info(int id) const { return shells_[id]; }

    // Degree distribution stats
    void degree_stats(float& mean, float& stddev, float& gini) const;

    // Save/load
    void save(const std::string& path) const;
    void load(const std::string& path);

    // Intrinsic dimensionality estimate (used to auto-set m)
    float estimate_intrinsic_dim(int sample = 2000) const;

    // Statistics logged during last search batch
    mutable float last_avg_candidates = 0.f;
    mutable float last_avg_hops       = 0.f;

private:
    // ---- internal state ----
    int n_ = 0, dim_ = 0, R_ = 0;
    float m_    = 2.0f;
    float alpha_ = 0.5f;
    int   start_node_ = 0;

    std::vector<float>            data_;    // n × dim, 64-byte aligned copy
    std::vector<std::vector<int>> adj_;     // adjacency list
    std::vector<ShellInfo>        shells_;  // per-node shell geometry
    std::vector<std::mutex>       mtx_;     // per-node mutex for parallel build

    // ---- internal methods ----

    // Greedy best-first search: returns sorted candidate list (up to L)
    std::vector<Candidate> greedy_search_internal(
        const float* query, int start, int L,
        int* hops_out = nullptr) const;

    // Radial-layer pruning: the key new algorithm
    // Given a candidate list for node p, select at most R neighbors
    // distributed across geometric shells, with angular diversity per shell.
    std::vector<int> radial_prune(
        int p,
        const std::vector<Candidate>& candidates) const;

    // Compute shell boundaries for node p given its nearest-neighbor distance
    ShellInfo compute_shells(float r_base) const;

    // Assign a candidate to a shell index (returns -1 if beyond all shells)
    int shell_index(const ShellInfo& si, float dist) const;

    // Budget per shell: allocate R slots across K shells
    // Inner shells get more slots (local refinement matters more)
    std::vector<int> shell_budget(const ShellInfo& si, int R_total,
                                  int non_empty_shells) const;

    // Select start node: medoid (point closest to centroid)
    int compute_medoid(int sample = 1000) const;

    // Build one pass
    void build_pass(int L_build);

    const float* node_data(int id) const {
        return data_.data() + (size_t)id * dim_;
    }
};

} // namespace rlg
