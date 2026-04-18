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
// Medoid Computation  (Algorithm 3: "let s denote the medoid of dataset P")
// ============================================================================
// The paper uses the dataset medoid — the most central point — as start_node_,
// rather than a random point. Starting from a central node ensures GreedySearch
// can reach any region of the vector space in fewer hops, reducing both build
// time (better candidates found early) and query latency.
//
// Exact medoid requires O(n^2) distance computations. We approximate it by:
//   1. Computing the centroid (mean vector) in double precision.
//   2. Finding the dataset point nearest to that centroid.
// This is the standard approximation used by DiskANN in practice.

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
// Random R-Regular Graph Initialization  (Algorithm 3, line 1)
// ============================================================================
// "Initialize G to a random R-regular directed graph."
//
// The original code started with an empty graph, which means early insertions
// have very few candidates to prune from and the search quality during the
// first pass is poor. Starting from a random R-regular graph guarantees global
// connectivity from the first insertion, giving GreedySearch a richer candidate
// set throughout the entire first pass and producing a higher-quality result.
//
// We assign each node exactly R distinct out-neighbors chosen uniformly at
// random, avoiding self-loops.

void VamanaIndex::init_random_graph(uint32_t R, std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> dist(0, npts_ - 1);
    for (uint32_t i = 0; i < npts_; i++) {
        std::set<uint32_t> chosen;
        while (chosen.size() < R) {
            uint32_t j = dist(rng);
            if (j != i)
                chosen.insert(j);
        }
        graph_[i].assign(chosen.begin(), chosen.end());
    }
}

// ============================================================================
// Greedy Search  (Algorithm 1 in the paper)
// ============================================================================
// Standard single-node-per-round best-first traversal. Each iteration finds
// the closest un-expanded candidate and evaluates all of its neighbors.
// The candidate set is bounded to L entries at all times.

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L) const {
    std::set<Candidate> candidate_set;
    std::vector<bool> visited(npts_, false);
    uint32_t dist_cmps = 0;

    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    candidate_set.insert({start_dist, start_node_});
    visited[start_node_] = true;

    std::set<uint32_t> expanded;

    while (true) {
        // Find the closest candidate not yet expanded
        uint32_t best_node = UINT32_MAX;
        for (const auto& [dist, id] : candidate_set) {
            if (expanded.find(id) == expanded.end()) {
                best_node = id;
                break;
            }
        }
        if (best_node == UINT32_MAX)
            break;

        expanded.insert(best_node);

        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }
        for (uint32_t nbr : neighbors) {
            if (visited[nbr]) continue;
            visited[nbr] = true;

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            if (candidate_set.size() < L) {
                candidate_set.insert({d, nbr});
            } else {
                auto worst = std::prev(candidate_set.end());
                if (d < worst->first) {
                    candidate_set.erase(worst);
                    candidate_set.insert({d, nbr});
                }
            }
        }
    }

    std::vector<Candidate> results(candidate_set.begin(), candidate_set.end());
    return {results, dist_cmps};
}

// ============================================================================
// BeamSearch  (Section 3.3: "DiskANN Beam Search")
// ============================================================================
// On an SSD, issuing W random reads in one shot costs roughly the same as
// issuing a single read (all reads are submitted together in one I/O round).
// BeamSearch exploits this by expanding the W closest un-expanded candidates
// per round instead of just 1, reducing the total number of I/O round-trips
// (hops) by a factor of up to W.
//
// For in-memory usage, the benefit is lower (no disk latency), but the reduced
// number of sequential loop iterations can still help on large graphs. The
// paper uses W = 4 or 8 for SSD serving; W = 1 degenerates to GreedySearch.
//
// The search list L is still bounded to L entries after each expansion round,
// just as in Algorithm 1.

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::beam_search(const float* query, uint32_t L, uint32_t W) const {
    if (W <= 1)
        return greedy_search(query, L);

    std::set<Candidate> candidate_set;
    std::vector<bool> visited(npts_, false);
    uint32_t dist_cmps = 0;

    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    candidate_set.insert({start_dist, start_node_});
    visited[start_node_] = true;

    std::set<uint32_t> expanded;

    while (true) {
        // Collect the W closest un-expanded candidates (the "beam")
        std::vector<uint32_t> beam;
        beam.reserve(W);
        for (const auto& [dist, id] : candidate_set) {
            if (expanded.find(id) == expanded.end()) {
                beam.push_back(id);
                if (beam.size() == W) break;
            }
        }
        if (beam.empty()) break;

        // Mark all beam nodes expanded before processing neighbors.
        // This mirrors issuing W parallel async disk reads in a single I/O round.
        for (uint32_t node : beam)
            expanded.insert(node);

        // Evaluate all neighbors of every beam node
        for (uint32_t node : beam) {
            std::vector<uint32_t> neighbors;
            {
                std::lock_guard<std::mutex> lock(locks_[node]);
                neighbors = graph_[node];
            }
            for (uint32_t nbr : neighbors) {
                if (visited[nbr]) continue;
                visited[nbr] = true;

                float d = compute_l2sq(query, get_vector(nbr), dim_);
                dist_cmps++;

                if (candidate_set.size() < L) {
                    candidate_set.insert({d, nbr});
                } else {
                    auto worst = std::prev(candidate_set.end());
                    if (d < worst->first) {
                        candidate_set.erase(worst);
                        candidate_set.insert({d, nbr});
                    }
                }
            }
        }
    }

    std::vector<Candidate> results(candidate_set.begin(), candidate_set.end());
    return {results, dist_cmps};
}

// ============================================================================
// Robust Prune  (Algorithm 2 in the paper)
// ============================================================================
// Greedy alpha-RNG pruning: keep at most R neighbors such that no chosen
// neighbor is "occluded" by a closer one (scaled by alpha).
//
// KEY FIX vs original code — Algorithm 2, line 1:
//   V ← (V ∪ Nout(p)) \ {p}
//
// The original implementation did NOT union in the node's existing out-neighbors
// before pruning. This matters when a backward edge triggers a re-prune of a
// neighbor: without including Nout(p), previously chosen good neighbors are
// silently dropped and cannot compete with newly added candidates. Including
// them ensures re-pruning is always a refinement, never a regression.
//
// Concurrency note: robust_prune is called under the node's lock from
// run_build_pass for the neighbor re-prune path (graph_[nbr] is cleared first
// then we call robust_prune which writes via its own internal lock). For the
// primary prune of `point` (which only this OMP thread writes), no external
// lock is needed. robust_prune acquires locks internally for both reads and
// writes to graph_[node].

void VamanaIndex::robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    // Union in existing out-neighbors (Algorithm 2, line 1: V ← V ∪ Nout(p))
    {
        std::lock_guard<std::mutex> lock(locks_[node]);
        for (uint32_t existing_nbr : graph_[node]) {
            if (existing_nbr == node) continue;
            float d = compute_l2sq(get_vector(node), get_vector(existing_nbr), dim_);
            candidates.push_back({d, existing_nbr});
        }
    }

    // Sort by id to deduplicate, keeping the minimum recorded distance per id
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.second < b.second ||
                         (a.second == b.second && a.first < b.first);
              });
    candidates.erase(
        std::unique(candidates.begin(), candidates.end(),
                    [](const Candidate& a, const Candidate& b) {
                        return a.second == b.second;
                    }),
        candidates.end());

    // Remove self
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());

    // Sort ascending by distance to node for the greedy selection pass
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R) break;

        // Alpha-RNG occlusion check: discard cand_id if some already-selected
        // neighbor n satisfies  alpha * d(cand_id, n) <= d(node, cand_id),
        // meaning n is "on the way" to cand_id from node.
        bool keep = true;
        for (uint32_t selected : new_neighbors) {
            float dist_cand_to_selected =
                compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
            if (alpha * dist_cand_to_selected <= dist_to_node) {
                keep = false;
                break;
            }
        }
        if (keep)
            new_neighbors.push_back(cand_id);
    }

    // Write the new neighbor list under lock
    {
        std::lock_guard<std::mutex> lock(locks_[node]);
        graph_[node] = std::move(new_neighbors);
    }
}

// ============================================================================
// Single Build Pass  (called twice by build())
// ============================================================================
// Iterates over all points in the given permutation order, searching for each
// point's approximate neighborhood in the current graph, pruning, and adding
// backward edges. The alpha parameter controls the pruning aggressiveness.

void VamanaIndex::run_build_pass(const std::vector<uint32_t>& perm,
                                 uint32_t R, uint32_t L,
                                 float alpha, uint32_t gamma_R) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        // Step 1: Find candidate neighbors via GreedySearch.
        // We always use W=1 (greedy) during build — BeamSearch is a
        // search-time optimization for SSD serving, not a build-time one.
        auto [candidates, _dist_cmps] = greedy_search(get_vector(point), L);

        // Step 2: Prune candidates to select this point's out-neighbors.
        // robust_prune reads & writes graph_[point] under its internal lock.
        robust_prune(point, candidates, alpha, R);

        // Step 3: Add backward edges and conditionally re-prune neighbors.
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
                // Build the candidate list from nbr's current neighbors,
                // then clear so robust_prune writes the pruned list fresh.
                std::vector<Candidate> nbr_candidates;
                {
                    std::lock_guard<std::mutex> lock(locks_[nbr]);
                    nbr_candidates.reserve(graph_[nbr].size());
                    for (uint32_t nn : graph_[nbr]) {
                        float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                        nbr_candidates.push_back({d, nn});
                    }
                    graph_[nbr].clear();  // robust_prune will repopulate
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
// Build  — Two-Pass Vamana  (Algorithm 3 + Section 2.3)
// ============================================================================
// The paper makes two passes over the dataset:
//
//   Pass 1 (alpha = 1.0):
//     Produces a well-connected base graph quickly. With alpha=1 the pruning
//     condition is the strict RNG property, so only genuinely short-range
//     neighbors survive. This gives a compact graph with low average degree
//     that GreedySearch can navigate.
//
//   Pass 2 (alpha = user value >= 1):
//     Relaxes pruning to allow long-range "highway" edges. A candidate c
//     that was dominated by a closer neighbor n in pass 1 may now survive if
//     alpha * d(c, n) > d(node, c). These extra edges reduce graph diameter
//     substantially, leading to fewer hops per query — critical for SSD
//     serving where each hop is a disk read.
//
// Running both passes at the user-defined alpha makes pass 1 slower (higher
// average degree) with no quality benefit, which is why the paper locks pass 1
// at alpha=1.

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
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

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    // Optimization 1: Medoid start node (Algorithm 3, "let s = medoid of P")
    std::cout << "Computing medoid start node..." << std::endl;
    start_node_ = compute_medoid();
    std::cout << "  Start node (medoid): " << start_node_ << std::endl;

    std::mt19937 rng(42);

    // Optimization 2: Random R-regular initialization (Algorithm 3, line 1)
    std::cout << "Initializing random R-regular graph..." << std::endl;
    init_random_graph(R, rng);

    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    std::cout << "Building index (R=" << R << ", L=" << L
              << ", alpha=" << alpha << ", gamma=" << gamma
              << ", gammaR=" << gamma_R << ")..." << std::endl;

    Timer build_timer;

    // Optimization 3: Two-pass build (Section 2.3)
    std::cout << "\n--- Pass 1 (alpha=1.0) ---" << std::endl;
    run_build_pass(perm, R, L, /*alpha=*/1.0f, gamma_R);
    std::cout << "\n  Pass 1 complete." << std::endl;

    std::shuffle(perm.begin(), perm.end(), rng);  // re-shuffle for pass 2

    std::cout << "\n--- Pass 2 (alpha=" << alpha << ") ---" << std::endl;
    run_build_pass(perm, R, L, alpha, gamma_R);
    std::cout << "\n  Pass 2 complete." << std::endl;

    double build_time = build_timer.elapsed_seconds();

    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++)
        total_edges += graph_[i].size();
    double avg_degree = (double)total_edges / npts_;

    std::cout << "\n  Build complete in " << build_time << " seconds."
              << std::endl;
    std::cout << "  Average out-degree: " << avg_degree << std::endl;
}

// ============================================================================
// Search
// ============================================================================
// beam_width = 1 → standard GreedySearch (default, fastest in-memory).
// beam_width > 1 → BeamSearch (fewer rounds; recommended for SSD, W=4 or 8).

SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L,
                                 uint32_t beam_width) const {
    if (L < K) L = K;

    Timer t;
    auto [candidates, dist_cmps] = beam_search(query, L, beam_width);
    double latency = t.elapsed_us();

    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++)
        result.ids.push_back(candidates[i].second);
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
        if (deg > 0)
            out.write(reinterpret_cast<const char*>(graph_[i].data()),
                      deg * sizeof(uint32_t));
    }

    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndex::load(const std::string& index_path,
                       const std::string& data_path) {
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

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
        if (deg > 0)
            in.read(reinterpret_cast<char*>(graph_[i].data()),
                    deg * sizeof(uint32_t));
    }

    std::cout << "Index loaded: " << npts_ << " points, " << dim_
              << " dims, start=" << start_node_ << std::endl;
}