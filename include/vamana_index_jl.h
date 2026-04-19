#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <string>
#include "jl_projection.h"

struct SearchResult {
    std::vector<uint32_t> ids;
    uint32_t dist_cmps;
    double latency_us;
};

// Vamana index with optional Johnson-Lindenstrauss dimensionality reduction.
// All distance computations run in the projected space (k dimensions instead
// of dim), giving up to (dim/k)x speedup on every distance call in both
// build and search, with approximate distance preservation guaranteed by the
// JL lemma.
class VamanaIndexJL {
  public:
    VamanaIndexJL() = default;
    ~VamanaIndexJL();

    // Build the graph.
    // proj_dim: target dimension after projection (0 = disabled, use full dim).
    //           Recommended starting point: dim/4 (e.g. 32 for SIFT1M).
    // L_build:  search list size during construction. Should be much larger
    //           than L used at query time (e.g. 200-500).
    void build(const std::string& data_path, uint32_t R, uint32_t L_build,
               float alpha, float gamma, uint32_t proj_dim = 0);

    // Search. L is the query-time search list, independent of L_build.
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    // Save writes index.bin + index.bin.proj (projection matrix).
    // Load reads both automatically.
    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    uint32_t get_npts()     const { return npts_; }
    uint32_t get_dim()      const { return dim_;  }
    uint32_t get_proj_dim() const { return proj_active_ ? proj_.k() : dim_; }

  private:
    using Candidate = std::pair<float, uint32_t>;

    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search(const float* query, uint32_t L,
                  const std::vector<uint32_t>& multi_starts = {}) const;

    void robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                      float alpha, uint32_t R);

    void run_build_pass(const std::vector<uint32_t>& perm,
                        uint32_t R, uint32_t L_build,
                        float alpha, uint32_t gamma_R);

    uint32_t compute_medoid() const;

    // Returns pointer into proj_data_ (k-dim) or data_ (full dim).
    const float* get_vector(uint32_t id) const {
        if (proj_active_)
            return proj_data_.data() + (size_t)id * proj_.k();
        return data_ + (size_t)id * dim_;
    }

    uint32_t working_dim() const {
        return proj_active_ ? proj_.k() : dim_;
    }

    // Original data
    float*   data_      = nullptr;
    uint32_t npts_      = 0;
    uint32_t dim_       = 0;
    bool     owns_data_ = false;

    // JL projection
    bool               proj_active_ = false;
    JLProjection       proj_;
    std::vector<float> proj_data_;   // projected dataset [npts x k]

    // Graph
    std::vector<std::vector<uint32_t>> graph_;
    uint32_t start_node_ = 0;

    mutable std::vector<std::mutex> locks_;
};