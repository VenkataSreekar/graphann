#pragma once
// rlg_index.h — Radial Layer Graph  (place in graphann/include/)
//
// ALGORITHM — Augmented Robust Prune
// ─────────────────────────────────────────────────────────────────────────────
// Identical to Vamana EXCEPT the neighbor budget is split:
//
//   RNG_LIMIT  = R - SHELL_SLOTS   (default: 24 of 32)  ← alpha-RNG neighbours
//   SHELL_SLOTS = R / 4            (default:  8 of 32)  ← one per distance shell
//
//   Stage 1: standard alpha-RNG up to RNG_LIMIT  → Vamana-quality local graph
//   Stage 2: fill remaining SHELL_SLOTS with the nearest unrepresented shell
//            shell k = [m^(k-1)*r_base,  m^k*r_base)   where r_base = nearest
//            Stage-1 neighbour distance
//
// This ALWAYS activates Stage 2 (unlike the old "if budget left" version).
// The shell neighbours give explicit multi-scale coverage per Kleinberg (2000).
//
// ALPHA is the standard Vamana RNG alpha (1.0–1.5), NOT an angular threshold.
// Use the same value as build_index: --alpha 1.2.

#include "io_utils.h"
#include "timer.h"

#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

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
    // Same parameters as build_index (R, L, alpha, gamma) + shell multiplier m.
    //   alpha : RNG diversity factor, 1.0–1.5  (same as Vamana --alpha)
    //   m     : geometric shell ratio,  1.5–3.0 (RLG-specific, default 2.0)
    //   gamma : degree-overflow multiplier (same as Vamana --gamma)
    void build(const std::string& data_path,
               uint32_t R     = 32,
               uint32_t L     = 75,
               float    alpha = 1.2f,
               float    m     = 2.0f,
               float    gamma = 1.5f);

    // ── Search ───────────────────────────────────────────────────────────────
    // Identical signature to VamanaIndex::search.
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    // ── Persistence ──────────────────────────────────────────────────────────
    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    // ── Accessors ────────────────────────────────────────────────────────────
    uint32_t get_npts() const { return npts_; }
    uint32_t get_dim()  const { return dim_;  }

    void degree_stats(float& mean, float& stddev, float& gini) const;

private:
    using Candidate = std::pair<float, uint32_t>;  // (dist, id) — same as Vamana

    // ── Data ─────────────────────────────────────────────────────────────────
    float*   data_      = nullptr;
    uint32_t npts_      = 0;
    uint32_t dim_       = 0;
    bool     owns_data_ = false;

    // ── Graph ────────────────────────────────────────────────────────────────
    std::vector<std::vector<uint32_t>> graph_;
    uint32_t start_node_ = 0;
    uint32_t R_          = 32;
    float    m_          = 2.0f;

    // ── Concurrency ──────────────────────────────────────────────────────────
    mutable std::vector<std::mutex> locks_;

    // ── Core algorithms ──────────────────────────────────────────────────────
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search(const float* query, uint32_t L) const;

    void augmented_robust_prune(uint32_t node,
                                 std::vector<Candidate>& cands,
                                 float alpha);

    void run_build_pass(const std::vector<uint32_t>& perm,
                        uint32_t R, uint32_t L,
                        float alpha, uint32_t gamma_R);

    int  dist_to_shell(float d, float r_base) const;

    uint32_t compute_medoid() const;

    const float* get_vec(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }
};

} // namespace rlg