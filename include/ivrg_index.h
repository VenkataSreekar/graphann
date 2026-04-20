#pragma once
// ivrg_index.h — Inverted Voronoi Routing Graph
// Place in graphann/include/ alongside vamana_index.h and rlg_index.h
//
// ── Algorithm ───────────────────────────────────────────────────────────────
//
//  IVRG = Vamana graph + Inverted Voronoi Routing layer
//
//  Build Phase:
//    1. Construct full Vamana graph (identical to VamanaIndex::build)
//    2. Run k-means++ on a sample to find K cluster centroids
//    3. For each cluster, store its nearest point as representative
//    4. Store all K centroids + representatives
//
//  Query Phase:
//    1. Route: Find nprobe nearest cluster centroids → their representatives
//    2. Also always include the global medoid
//    3. Multi-seed greedy search from all seeds simultaneously
//    4. Return top K results
//
//  Result: better initial seeds → lower search cost without sacrificing quality.
//
// ── Interface ────────────────────────────────────────────────────────────────
//  Mirrors VamanaIndex so search_ivrg.cpp ≈ search_index.cpp, but with
//  additional parameters K (clusters), nprobe (seeds per query), T (k-means iters).

#include "io_utils.h"   // FloatMatrix, IntMatrix, AlignedFree, load_fbin
#include "timer.h"      // Timer

#include <cstdint>
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
    // Parameters extend VamanaIndex with routing-layer config:
    //   data_path   : .fbin file  (same format as Vamana)
    //   R           : max out-degree per node
    //   L           : beam width during construction  (L >= R)
    //   alpha       : α-RNG diversity factor  (1.0–1.5, same as Vamana --alpha)
    //   gamma       : over-degree trigger = gamma*R  (same as Vamana --gamma)
    //   K_clusters  : number of Voronoi clusters  (e.g. 512)
    //   nprobe      : number of cluster seeds per query  (e.g. 3)
    //   T_iter      : Lloyd's iterations for k-means  (e.g. 15)
    //   kmeans_sample : sample size for k-means  (e.g. 100000)
    void build(const std::string& data_path,
               uint32_t R     = 32,
               uint32_t L     = 75,
               float    alpha = 1.2f,
               float    gamma = 1.5f,
               uint32_t K_clusters   = 512,
               uint32_t nprobe       = 3,
               uint32_t T_iter       = 15,
               uint32_t kmeans_sample = 100000);

    // ── Search ───────────────────────────────────────────────────────────────
    // Identical signature to VamanaIndex::search so search_ivrg.cpp
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
    greedy_search(const float* query, uint32_t L,
                  const std::vector<uint32_t>& seeds) const;

    // Single-seed version used during build
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search_from(const float* query, uint32_t start, uint32_t L) const;

    // Routing layer: find nprobe nearest cluster centroids and return reps
    std::vector<uint32_t> route(const float* query, uint32_t np) const;

    // Build the routing layer: k-means + representative selection
    void build_routing_layer(uint32_t K,
                              uint32_t T_iter,
                              uint32_t kmeans_sample);

    // Alpha-RNG pruning (identical to Vamana)
    void robust_prune(uint32_t node,
                      std::vector<Candidate>& candidates,
                      float alpha, uint32_t R);

    // One full randomised pass over all points (identical to Vamana)
    void run_build_pass(const std::vector<uint32_t>& perm,
                        uint32_t R, uint32_t L,
                        float alpha, uint32_t gamma_R);

    // ── Data ─────────────────────────────────────────────────────────────────
    float*   data_      = nullptr;
    uint32_t npts_      = 0;
    uint32_t dim_       = 0;
    bool     owns_data_ = false;

    // ── Graph ────────────────────────────────────────────────────────────────
    std::vector<std::vector<uint32_t>> graph_;
    uint32_t start_node_ = 0;

    // ── Routing layer ────────────────────────────────────────────────────────
    uint32_t K_       = 512;    // number of clusters
    uint32_t nprobe_  = 3;      // seeds per query
    std::vector<std::vector<float>> centroids_;      // K_ cluster centers
    std::vector<uint32_t>           representatives_; // K_ nearest points

    // ── Concurrency ──────────────────────────────────────────────────────────
    mutable std::vector<std::mutex> locks_;

    // ── Helpers ──────────────────────────────────────────────────────────────
    uint32_t compute_medoid() const;

    const float* get_vec(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }
};

} // namespace ivrg
