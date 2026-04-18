#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <string>
#include <random>  // for std::mt19937 used in init_random_graph

// Result of a single query search.
struct SearchResult {
    std::vector<uint32_t> ids;  // nearest neighbor IDs (sorted by distance)
    uint32_t dist_cmps;         // number of distance computations
    double latency_us;          // search latency in microseconds
};

// Vamana graph-based approximate nearest neighbor index.
//
// Key concepts:
//   - Build makes TWO passes: pass 1 with alpha=1.0, pass 2 with user alpha.
//   - Graph is initialized as a random R-regular directed graph (not empty).
//   - start_node_ is the dataset medoid, not a random node.
//   - BeamSearch expands W nodes per round instead of 1 (W=4-8 for SSD).
//   - robust_prune unions existing Nout(p) into candidates (Algorithm 2, line 1).
class VamanaIndex {
  public:
    VamanaIndex() = default;
    ~VamanaIndex();

    // Build the Vamana graph in two passes (pass 1 always uses alpha=1.0).
    void build(const std::string& data_path, uint32_t R, uint32_t L,
               float alpha, float gamma);

    // Search for K nearest neighbors.
    // beam_width=1 -> GreedySearch; beam_width>1 -> BeamSearch (SSD: use 4-8).
    SearchResult search(const float* query, uint32_t K, uint32_t L,
                        uint32_t beam_width = 1) const;

    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    uint32_t get_npts() const { return npts_; }
    uint32_t get_dim()  const { return dim_; }

  private:
    using Candidate = std::pair<float, uint32_t>;

    // Core algorithms
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search(const float* query, uint32_t L) const;

    std::pair<std::vector<Candidate>, uint32_t>
    beam_search(const float* query, uint32_t L, uint32_t W) const;

    void robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                      float alpha, uint32_t R);

    void run_build_pass(const std::vector<uint32_t>& perm,
                        uint32_t R, uint32_t L,
                        float alpha, uint32_t gamma_R);

    uint32_t compute_medoid() const;
    void init_random_graph(uint32_t R, std::mt19937& rng);

    // Data
    float*   data_      = nullptr;
    uint32_t npts_      = 0;
    uint32_t dim_       = 0;
    bool     owns_data_ = false;

    // Graph
    std::vector<std::vector<uint32_t>> graph_;
    uint32_t start_node_ = 0;

    // Concurrency
    mutable std::vector<std::mutex> locks_;

    const float* get_vector(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }
};