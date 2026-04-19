#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <cstdlib>

// ============================================================================
// Destructor
// ============================================================================

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

// ============================================================================
// Greedy Search (Fixed Beam Width L + Multi-Start)
// ============================================================================
std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L, const std::vector<uint32_t>& multi_starts) const {
    std::set<Candidate> candidate_set;
    std::vector<bool> visited(npts_, false);
    uint32_t dist_cmps = 0;

    // 1. Seed with the primary start node (Medoid)
    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    candidate_set.insert({start_dist, start_node_});
    visited[start_node_] = true;

    // 2. --- INJECT MULTI-STARTS (To escape local minima) ---
    for (uint32_t random_start : multi_starts) {
        if (!visited[random_start]) {
            float d = compute_l2sq(query, get_vector(random_start), dim_);
            dist_cmps++;
            candidate_set.insert({d, random_start});
            visited[random_start] = true;
        }
    }
    
    // Ensure we don't accidentally exceed L right out of the gate
    while (candidate_set.size() > L) {
        candidate_set.erase(std::prev(candidate_set.end()));
    }
    // -------------------------------------------------------

    std::set<uint32_t> expanded;

    while (true) {
        // Find closest candidate that hasn't been expanded yet
        uint32_t best_node = UINT32_MAX;

        for (const auto& [dist, id] : candidate_set) {
            if (expanded.find(id) == expanded.end()) {
                best_node = id;
                break;
            }
        }
        if (best_node == UINT32_MAX)
            break;  // all candidates expanded

        expanded.insert(best_node);

        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }

        for (uint32_t nbr : neighbors) {
            if (visited[nbr])
                continue;
            visited[nbr] = true;

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            // --- FIXED L TRIMMING ---
            // Insert if candidate set isn't full or this is closer than worst
            if (candidate_set.size() < L) {
                candidate_set.insert({d, nbr});
            } else {
                auto worst = std::prev(candidate_set.end());
                if (d < worst->first) {
                    candidate_set.erase(worst);
                    candidate_set.insert({d, nbr});
                }
            }
            // ------------------------
        }
    }

    // Convert to sorted vector
    std::vector<Candidate> results(candidate_set.begin(), candidate_set.end());
    return {results, dist_cmps};
}

// ============================================================================
// Robust Prune (Alpha-RNG Rule)
// ============================================================================
void VamanaIndex::robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    // Remove self from candidates if present
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());

    // Sort by distance to node (ascending)
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R)
            break;

        // Check alpha-RNG condition against all already-selected neighbors
        bool keep = true;
        for (uint32_t selected : new_neighbors) {
            float dist_cand_to_selected =
                compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
            if (dist_to_node > alpha * dist_cand_to_selected) {
                keep = false;
                break;
            }
        }

        if (keep)
            new_neighbors.push_back(cand_id);
    }

    // Thread-safe update of the node's neighborhood
    {
        std::lock_guard<std::mutex> lock(locks_[node]);
        graph_[node] = std::move(new_neighbors);
    }
}

// ============================================================================
// Medoid Computation
// ============================================================================
uint32_t VamanaIndex::compute_medoid() const {
    // Step 1: accumulate centroid
    std::vector<double> centroid(dim_, 0.0);
    for (uint32_t i = 0; i < npts_; i++) {
        const float* vec = get_vector(i);
        for (uint32_t d = 0; d < dim_; d++)
            centroid[d] += vec[d];
    }
    for (uint32_t d = 0; d < dim_; d++)
        centroid[d] /= npts_;

    // Step 2: find closest point to centroid
    uint32_t medoid = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < npts_; i++) {
        float dist = 0.0f;
        const float* vec = get_vector(i);
        for (uint32_t d = 0; d < dim_; d++) {
            float diff = vec[d] - static_cast<float>(centroid[d]);
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            medoid = i;
        }
    }
    return medoid;
}

// ============================================================================
// Build (Two-Pass)
// ============================================================================
void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    // --- Load data ---
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dimensions: " << dim_ << std::endl;

    if (L < R) {
        std::cerr << "Warning: L (" << L << ") < R (" << R
                  << "). Setting L = R." << std::endl;
        L = R;
    }

    // --- Initialize empty graph and per-node locks ---
    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    // --- Pick medoid start node ---
    std::mt19937 rng(42);  // fixed seed for reproducibility
    start_node_ = compute_medoid();
    std::cout << "  Start node (medoid): " << start_node_ << std::endl;

    // --- Create random insertion order ---
    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    std::cout << "Building index (R=" << R << ", L=" << L
              << ", alpha=" << alpha << ", gamma=" << gamma
              << ", gammaR=" << gamma_R << ")..." << std::endl;

    Timer build_timer;

    // --- Pass 1: Build base graph with strict alpha=1.0 ---
    std::cout << "\n--- Pass 1 (alpha=1.0) ---" << std::endl;
    run_build_pass(perm, R, L, 1.0f, gamma_R);
    std::cout << "\n  Pass 1 complete." << std::endl;

    // --- Pass 2: Refine graph with long-range edges using user alpha ---
    std::cout << "\n--- Pass 2 (alpha=" << alpha << ") ---" << std::endl;
    std::shuffle(perm.begin(), perm.end(), rng); // re-shuffle for Pass 2
    run_build_pass(perm, R, L, alpha, gamma_R);
    std::cout << "\n  Pass 2 complete." << std::endl;

    double build_time = build_timer.elapsed_seconds();

    // Compute average degree
    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++)
        total_edges += graph_[i].size();
    double avg_degree = (double)total_edges / npts_;

    std::cout << "\n  Build complete in " << build_time << " seconds."
              << std::endl;
    std::cout << "  Average out-degree: " << avg_degree << std::endl;
}

// ============================================================================
// Single Build Pass Helper
// ============================================================================
void VamanaIndex::run_build_pass(const std::vector<uint32_t>& perm,
                                 uint32_t R, uint32_t L,
                                 float alpha, uint32_t gamma_R) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        // Step 1: Find candidate neighbors via GreedySearch
        auto [candidates, _dist_cmps] = greedy_search(get_vector(point), L);

        // --- NEW: NEIGHBORHOOD MERGING (Fixes Catastrophic Forgetting) ---
        // Ensure that existing edges (from Pass 1 or backward inserts) aren't discarded
        {
            std::lock_guard<std::mutex> lock(locks_[point]);
            for (uint32_t existing_nbr : graph_[point]) {
                float d = compute_l2sq(get_vector(point), get_vector(existing_nbr), dim_);
                candidates.push_back({d, existing_nbr});
            }
        }
        // -----------------------------------------------------------------

        // Step 2: Prune candidates to select out-neighbors
        robust_prune(point, candidates, alpha, R);

        // Step 3: Add backward edges and conditionally re-prune
        std::vector<uint32_t> my_neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[point]);
            my_neighbors = graph_[point];
        }

        for (uint32_t nbr : my_neighbors) {
            bool needs_prune = false;
            {
                std::lock_guard<std::mutex> lock(locks_[nbr]);
                graph_[nbr].push_back(point);
                needs_prune = (graph_[nbr].size() > gamma_R);
            }

            if (needs_prune) {
                std::vector<Candidate> nbr_candidates;
                {
                    std::lock_guard<std::mutex> lock(locks_[nbr]);
                    nbr_candidates.reserve(graph_[nbr].size());
                    for (uint32_t nn : graph_[nbr]) {
                        float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                        nbr_candidates.push_back({d, nn});
                    }
                }
                robust_prune(nbr, nbr_candidates, alpha, R);
            }
        }

        if (idx % 10000 == 0) {
            #pragma omp critical
            {
                std::cout << "\r  Processed " << idx << " / " << npts_
                          << " points" << std::flush;
            }
        }
    }
}

// ============================================================================
// Search
// ============================================================================
SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L) const {
    if (L < K) L = K;

    Timer t;

    // --- GENERATE MULTI-START ENSEMBLE ---
    // Pick 3 random points to act as alternative starting lines.
    // This attacks the graph from 4 total angles (Medoid + 3 Randoms).
    std::vector<uint32_t> ensemble_starts;
    for(int i = 0; i < 3; i++) {
        // Prevent modulo bias or zero-division edge cases
        if (npts_ > 0) {
            ensemble_starts.push_back(std::rand() % npts_);
        }
    }
    // -------------------------------------

    auto [candidates, dist_cmps] = greedy_search(query, L, ensemble_starts);
    double latency = t.elapsed_us();

    // Return top-K results
    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

// ============================================================================
// Save / Load
// ============================================================================

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open file for writing: " + path);

    out.write(reinterpret_cast<const char*>(&npts_), 4);
    out.write(reinterpret_cast<const char*>(&dim_), 4);
    out.write(reinterpret_cast<const char*>(&start_node_), 4);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = graph_[i].size();
        out.write(reinterpret_cast<const char*>(&deg), 4);
        if (deg > 0) {
            out.write(reinterpret_cast<const char*>(graph_[i].data()),
                      deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndex::load(const std::string& index_path,
                       const std::string& data_path) {
    // Load data vectors
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    // Load graph
    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open index file: " + index_path);

    uint32_t file_npts, file_dim;
    in.read(reinterpret_cast<char*>(&file_npts), 4);
    in.read(reinterpret_cast<char*>(&file_dim), 4);
    in.read(reinterpret_cast<char*>(&start_node_), 4);

    if (file_npts != npts_ || file_dim != dim_)
        throw std::runtime_error(
            "Index/data mismatch: index has " + std::to_string(file_npts) +
            "x" + std::to_string(file_dim) + ", data has " +
            std::to_string(npts_) + "x" + std::to_string(dim_));

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), 4);
        graph_[i].resize(deg);
        if (deg > 0) {
            in.read(reinterpret_cast<char*>(graph_[i].data()),
                    deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index loaded: " << npts_ << " points, " << dim_
              << " dims, start=" << start_node_ << std::endl;
}