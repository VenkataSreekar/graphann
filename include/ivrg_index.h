#pragma once
// ivrg_index.h — Inverted Voronoi Routing Graph
// Place in graphann/include/
//
// ALGORITHM
// ─────────────────────────────────────────────────────────────────────────────
// The graph (edges, R, alpha, gamma) is IDENTICAL to a standard Vamana build.
// The ONLY difference is how search is seeded:
//
//   Vamana search:  always starts from 1 fixed global medoid
//   IVRG   search:  starts from nprobe data points closest to the query's
//                   Voronoi cell centroids  +  global medoid as safety fallback
//
// This eliminates the "travel hops" Vamana spends getting from the medoid to
// the query's neighborhood, reducing Avg Dist Cmps at the same recall level.
//
// BUILD OVERHEAD vs VAMANA
//   Vamana graph build : ~250 s  (same, no change)
//   k-means (K=512)    :  ~15 s  (Lloyd on 200K sample)
//   Representative find:   ~2 s  (O(S×K), done during final Lloyd pass)
//
// PARAMETERS
//   R, L, alpha, gamma : identical to build_index.cpp (Vamana)
//   K_clusters         : Voronoi cells (256–1024, default 512)
//   nprobe             : seeds per query (1–8, default 3)
//   T_iter             : Lloyd's iterations (10–20, default 15)

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

    void build(const std::string& data_path,
               uint32_t R            = 32,
               uint32_t L            = 75,
               float    alpha        = 1.2f,
               float    gamma        = 1.5f,
               uint32_t K_clusters   = 512,
               uint32_t nprobe       = 3,
               uint32_t T_iter       = 15,
               uint32_t kmeans_sample= 200000);

    // Identical signature to VamanaIndex::search
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    uint32_t get_npts() const { return npts_; }
    uint32_t get_dim()  const { return dim_;  }
    uint32_t get_K()    const { return K_;    }

    void degree_stats(float& mean, float& stddev, float& gini) const;

private:
    using Candidate = std::pair<float, uint32_t>;

    float*   data_      = nullptr;
    uint32_t npts_      = 0;
    uint32_t dim_       = 0;
    bool     owns_data_ = false;

    std::vector<std::vector<uint32_t>> graph_;
    uint32_t start_node_ = 0;

    mutable std::vector<std::mutex> locks_;

    uint32_t K_      = 512;
    uint32_t nprobe_ = 3;

    std::vector<std::vector<float>> centroids_;      // K_ × dim_
    std::vector<uint32_t>           representatives_; // K_ nearest data points

    // Multi-seed greedy search (seed list can include medoid + routed points)
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

    const float* get_vec(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }
};

} // namespace ivrg