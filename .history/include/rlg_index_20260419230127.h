#pragma once
// rlg_index.h  — Radial Layer Graph index
// Drop this file into graphann/include/ alongside vamana_index.h
//
// Algorithm summary
// -----------------
// For every node p, anchor a set of geometric distance shells using the
// distance to p's nearest candidate as r_base:
//
//   Shell 0  :  [0,             r_base)
//   Shell 1  :  [r_base,        m*r_base)
//   Shell 2  :  [m*r_base,      m²*r_base)
//   Shell k  :  [m^(k-1)*r_base, m^k*r_base)
//
// Up to R neighbors are selected across shells using a harmonic slot budget
// (inner shells get more slots) and an angular diversity filter within each
// shell (two neighbors in the same shell must point in sufficiently different
// directions from p).
//
// Theoretical basis: Kleinberg (STOC 2000) proved that greedy routing on a
// graph achieves O(log² n) expected path length if edge lengths follow a
// geometric/power-law distribution — exactly what our shells produce.
//
// Search is identical to Vamana greedy beam search; the multi-scale edge
// structure is exploited automatically (long jumps early, local refine late).

#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace rlg {

// ── tiny helpers ────────────────────────────────────────────────────────────

inline float l2sq(const float* a, const float* b, int dim) {
    float s = 0.f;
    for (int i = 0; i < dim; ++i) { float d = a[i]-b[i]; s += d*d; }
    return s;
}

inline float dot(const float* a, const float* b, int dim) {
    float s = 0.f;
    for (int i = 0; i < dim; ++i) s += a[i]*b[i];
    return s;
}

// cosine similarity between two direction vectors
inline float cos_sim(const float* u, const float* v, int dim) {
    float uv = dot(u,v,dim), uu = dot(u,u,dim), vv = dot(v,v,dim);
    if (uu < 1e-12f || vv < 1e-12f) return 0.f;
    return uv / std::sqrt(uu * vv);
}

struct Candidate {
    float dist;
    int   id;
    bool operator<(const Candidate& o) const {
        return dist < o.dist || (dist == o.dist && id < o.id);
    }
};

// ── RLGIndex ────────────────────────────────────────────────────────────────

class RLGIndex {
public:
    // ------------------------------------------------------------------
    // build()  — construct the Radial Layer Graph
    //
    //   data      : row-major float array, n × dim
    //   n, dim    : dataset size and dimensionality
    //   R         : max degree per node  (same meaning as Vamana R)
    //   m         : shell multiplier — geometric ratio between layer radii
    //               typical range [1.3, 3.0].  Use m=2.0 as default.
    //               Theoretically optimal: m = 2^(1 / intrinsic_dim)
    //   L_build   : beam width for greedy search during construction
    //   alpha_ang : angular diversity threshold within a shell (cosine sim)
    //               0.0 = no filter (like Vamana),  0.9 = very strict
    //   two_pass  : run a second refinement pass to fix early-insertion bias
    // ------------------------------------------------------------------
    void build(const float* data, int n, int dim,
               int   R         = 32,
               float m         = 2.0f,
               int   L_build   = 75,
               float alpha_ang = 0.5f,
               bool  two_pass  = true);

    // ------------------------------------------------------------------
    // search()  — greedy beam search, identical to Vamana search
    //
    //   query     : float array of length dim
    //   K         : number of nearest neighbors to return
    //   L_search  : beam width (≥ K)
    //
    //   returns   : vector of K neighbor IDs sorted by distance
    // ------------------------------------------------------------------
    std::vector<int> search(const float* query, int K, int L_search) const;

    // save / load  (binary format, not compatible with Vamana .bin)
    void save(const std::string& path) const;
    void load(const std::string& path);

    // accessors used by build_rlg / search_rlg CLIs
    int   n()   const { return n_;   }
    int   dim() const { return dim_; }

    // degree distribution: mean, stddev, gini coefficient
    void degree_stats(float& mean, float& stddev, float& gini) const;

    // estimated intrinsic dimensionality (Levina-Bickel 2-NN estimator)
    float intrinsic_dim(int sample = 2000) const;

private:
    int   n_    = 0;
    int   dim_  = 0;
    int   R_    = 32;
    float m_    = 2.0f;
    float alpha_ang_ = 0.5f;
    int   start_node_ = 0;

    std::vector<float>            data_;   // n × dim  (owned copy)
    std::vector<std::vector<int>> adj_;    // adjacency lists
    mutable std::vector<std::mutex> mtx_;    // per-node mutex

    const float* node(int i) const { return data_.data() + (size_t)i * dim_; }

    // core internals
    std::vector<Candidate> greedy_search(const float* q, int start, int L) const;
    std::vector<int>       radial_prune (int p, std::vector<Candidate>& cands) const;
    int                    medoid       (int sample = 2000) const;
    void                   build_pass   (int L_build);
};

} // namespace rlg
