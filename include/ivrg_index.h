#pragma once

#include "io_utils.h"
#include "timer.h"

#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace ivrg {

struct SearchResult {
    std::vector<uint32_t> ids;
    uint32_t dist_cmps;
    double   latency_us;
};

class IVRGIndex {
public:
    IVRGIndex()  = default;
    ~IVRGIndex() { if (owns_data_ && data_) { std::free(data_); data_ = nullptr; } }

    // ── Build ────────────────────────────────────────────────────────────────
    void build(const std::string& data_path,
               uint32_t R            = 32,
               uint32_t L            = 75,
               float    alpha        = 1.2f,
               float    gamma        = 1.5f,
               uint32_t K_clusters   = 512,
               uint32_t nprobe       = 3,
               uint32_t T_iter       = 15,
               uint32_t kmeans_sample= 200000);

    // ── Search ───────────────────────────────────────────────────────────────
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    // ── Persistence ──────────────────────────────────────────────────────────
    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    // ── Accessors ────────────────────────────────────────────────────────────
    uint32_t get_npts() const { return npts_; }
    uint32_t get_dim()  const { return dim_;  }
    uint32_t get_K()    const { return K_;    }

    void degree_stats(float& mean, float& stddev, float& gini) const;

private:
    using Candidate = std::pair<float, uint32_t>;

    // ── Core algorithms ──────────────────────────────────────────────────────
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search(const float* query, uint32_t L,
                  const std::vector<uint32_t>& seeds) const;

    void robust_prune(uint32_t node, std::vector<Candidate>& cands,
                      float alpha, uint32_t R);

    void run_build_pass(const std::vector<uint32_t>& perm,
                        uint32_t R, uint32_t L,
                        float alpha, uint32_t gamma_R);

    void build_routing_layer(uint32_t K, uint32_t T_iter, uint32_t kmeans_sample);

    std::vector<uint32_t> route(const float* query, uint32_t np) const;

    uint32_t compute_medoid() const;

    // ── Data ─────────────────────────────────────────────────────────────────
    float* data_      = nullptr;
    uint32_t npts_      = 0;
    uint32_t dim_       = 0;
    bool     owns_data_ = false;

    // ── Graph ────────────────────────────────────────────────────────────────
    std::vector<std::vector<uint32_t>> graph_;
    uint32_t start_node_ = 0;

    mutable std::vector<std::mutex> locks_;

    // ── Routing layer ────────────────────────────────────────────────────────
    uint32_t K_      = 512;
    uint32_t nprobe_ = 3;

    std::vector<std::vector<float>> centroids_;
    std::vector<uint32_t>           representatives_;

    const float* get_vec(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }
};

} // namespace ivrg