#pragma once
// ivrg_index.h — Inverted Voronoi Routing Graph
<<<<<<< HEAD
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
=======
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
>>>>>>> 33f3e72cf903476ae9c3cb9b3b6d403bb7bbc8ff
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

<<<<<<< HEAD
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
=======
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

>>>>>>> 33f3e72cf903476ae9c3cb9b3b6d403bb7bbc8ff
    float*   data_      = nullptr;
    uint32_t npts_      = 0;
    uint32_t dim_       = 0;
    bool     owns_data_ = false;

<<<<<<< HEAD
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
=======
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

>>>>>>> 33f3e72cf903476ae9c3cb9b3b6d403bb7bbc8ff
    uint32_t compute_medoid() const;

    const float* get_vec(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }
};

<<<<<<< HEAD
} // namespace ivrg
=======
} // namespace ivrg
>>>>>>> 33f3e72cf903476ae9c3cb9b3b6d403bb7bbc8ff
