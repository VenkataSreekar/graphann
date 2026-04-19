#include "vamana_index_jl.h"
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

VamanaIndexJL::~VamanaIndexJL() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

// ============================================================================
// Greedy Search
// ============================================================================

std::pair<std::vector<VamanaIndexJL::Candidate>, uint32_t>
VamanaIndexJL::greedy_search(const float* query, uint32_t L,
                              const std::vector<uint32_t>& multi_starts) const {
    std::set<Candidate> candidate_set;
    std::vector<bool> visited(npts_, false);
    uint32_t dist_cmps = 0;
    uint32_t wdim = working_dim();

    float start_dist = compute_l2sq(query, get_vector(start_node_), wdim);
    dist_cmps++;
    candidate_set.insert({start_dist, start_node_});
    visited[start_node_] = true;

    for (uint32_t s : multi_starts) {
        if (!visited[s]) {
            float d = compute_l2sq(query, get_vector(s), wdim);
            dist_cmps++;
            candidate_set.insert({d, s});
            visited[s] = true;
        }
    }
    while (candidate_set.size() > L)
        candidate_set.erase(std::prev(candidate_set.end()));

    std::set<uint32_t> expanded;
    while (true) {
        uint32_t best_node = UINT32_MAX;
        for (const auto& [dist, id] : candidate_set) {
            if (!expanded.count(id)) { best_node = id; break; }
        }
        if (best_node == UINT32_MAX) break;
        expanded.insert(best_node);

        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }
        for (uint32_t nbr : neighbors) {
            if (visited[nbr]) continue;
            visited[nbr] = true;
            float d = compute_l2sq(query, get_vector(nbr), wdim);
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
    return {std::vector<Candidate>(candidate_set.begin(), candidate_set.end()),
            dist_cmps};
}

// ============================================================================
// Robust Prune
// ============================================================================

void VamanaIndexJL::robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                                  float alpha, uint32_t R) {
    uint32_t wdim = working_dim();
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R) break;
        bool keep = true;
        for (uint32_t sel : new_neighbors) {
            float d = compute_l2sq(get_vector(cand_id), get_vector(sel), wdim);
            if (dist_to_node > alpha * d) { keep = false; break; }
        }
        if (keep) new_neighbors.push_back(cand_id);
    }
    {
        std::lock_guard<std::mutex> lock(locks_[node]);
        graph_[node] = std::move(new_neighbors);
    }
}

// ============================================================================
// Medoid Computation
// ============================================================================

uint32_t VamanaIndexJL::compute_medoid() const {
    uint32_t wdim = working_dim();
    std::vector<double> centroid(wdim, 0.0);
    for (uint32_t i = 0; i < npts_; i++) {
        const float* v = get_vector(i);
        for (uint32_t d = 0; d < wdim; d++) centroid[d] += v[d];
    }
    for (uint32_t d = 0; d < wdim; d++) centroid[d] /= npts_;

    uint32_t medoid = 0;
    float best = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < npts_; i++) {
        float dist = 0.0f;
        const float* v = get_vector(i);
        for (uint32_t d = 0; d < wdim; d++) {
            float diff = v[d] - (float)centroid[d];
            dist += diff * diff;
        }
        if (dist < best) { best = dist; medoid = i; }
    }
    return medoid;
}

// ============================================================================
// Build
// ============================================================================

void VamanaIndexJL::build(const std::string& data_path, uint32_t R, uint32_t L_build,
                           float alpha, float gamma, uint32_t proj_dim) {
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;
    std::cout << "  Points: " << npts_ << ", Dimensions: " << dim_ << std::endl;

    if (proj_dim > 0 && proj_dim < dim_) {
        std::cout << "  Projecting " << dim_ << "D -> " << proj_dim
                  << "D via random orthonormal matrix..." << std::endl;
        proj_.init(dim_, proj_dim, /*seed=*/42);
        proj_data_ = proj_.project_dataset(data_, npts_);
        proj_active_ = true;
        std::cout << "  Projection complete. Distance calls now O(" << proj_dim
                  << ") instead of O(" << dim_ << ")." << std::endl;
    } else {
        proj_active_ = false;
    }

    if (L_build < R) {
        std::cerr << "Warning: L_build < R, setting L_build = R." << std::endl;
        L_build = R;
    }

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    std::mt19937 rng(42);
    start_node_ = compute_medoid();
    std::cout << "  Start node (medoid): " << start_node_ << std::endl;

    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    std::cout << "Building index (R=" << R << ", L_build=" << L_build
              << ", alpha=" << alpha << ", gamma=" << gamma
              << ", working_dim=" << working_dim() << ")..." << std::endl;

    Timer build_timer;

    std::cout << "\n--- Pass 1 (alpha=1.0) ---" << std::endl;
    run_build_pass(perm, R, L_build, 1.0f, gamma_R);
    std::cout << "\n  Pass 1 complete." << std::endl;

    std::shuffle(perm.begin(), perm.end(), rng);
    std::cout << "\n--- Pass 2 (alpha=" << alpha << ") ---" << std::endl;
    run_build_pass(perm, R, L_build, alpha, gamma_R);
    std::cout << "\n  Pass 2 complete." << std::endl;

    double build_time = build_timer.elapsed_seconds();
    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++) total_edges += graph_[i].size();
    std::cout << "\n  Build complete in " << build_time << " seconds." << std::endl;
    std::cout << "  Average out-degree: " << (double)total_edges / npts_ << std::endl;
}

// ============================================================================
// Single Build Pass
// ============================================================================

void VamanaIndexJL::run_build_pass(const std::vector<uint32_t>& perm,
                                    uint32_t R, uint32_t L_build,
                                    float alpha, uint32_t gamma_R) {
    uint32_t wdim = working_dim();

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];
        auto [candidates, _] = greedy_search(get_vector(point), L_build);

        {
            std::lock_guard<std::mutex> lock(locks_[point]);
            for (uint32_t nbr : graph_[point]) {
                float d = compute_l2sq(get_vector(point), get_vector(nbr), wdim);
                candidates.push_back({d, nbr});
            }
        }

        robust_prune(point, candidates, alpha, R);

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
                std::vector<Candidate> nbr_cands;
                {
                    std::lock_guard<std::mutex> lock(locks_[nbr]);
                    for (uint32_t nn : graph_[nbr]) {
                        float d = compute_l2sq(get_vector(nbr), get_vector(nn), wdim);
                        nbr_cands.push_back({d, nn});
                    }
                }
                robust_prune(nbr, nbr_cands, alpha, R);
            }
        }

        if (idx % 10000 == 0) {
            #pragma omp critical
            std::cout << "\r  Processed " << idx << " / " << npts_ << std::flush;
        }
    }
}

// ============================================================================
// Search
// ============================================================================

SearchResult VamanaIndexJL::search(const float* query, uint32_t K, uint32_t L) const {
    if (L < K) L = K;

    std::vector<float> proj_query;
    const float* q_ptr = query;
    if (proj_active_) {
        proj_query.resize(proj_.k());
        proj_.project(query, proj_query.data());
        q_ptr = proj_query.data();
    }

    Timer t;
    std::vector<uint32_t> ensemble_starts;
    for (int i = 0; i < 3; i++)
        if (npts_ > 0) ensemble_starts.push_back(std::rand() % npts_);

    auto [candidates, dist_cmps] = greedy_search(q_ptr, L, ensemble_starts);
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

void VamanaIndexJL::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open: " + path);

    out.write(reinterpret_cast<const char*>(&npts_),       4);
    out.write(reinterpret_cast<const char*>(&dim_),        4);
    out.write(reinterpret_cast<const char*>(&start_node_), 4);

    uint32_t proj_flag = proj_active_ ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&proj_flag), 4);
    if (proj_active_) {
        std::string proj_path = path + ".proj";
        proj_.save(proj_path);
        std::cout << "  Projection matrix saved to " << proj_path << std::endl;
    }

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = graph_[i].size();
        out.write(reinterpret_cast<const char*>(&deg), 4);
        if (deg > 0)
            out.write(reinterpret_cast<const char*>(graph_[i].data()),
                      deg * sizeof(uint32_t));
    }
    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndexJL::load(const std::string& index_path,
                          const std::string& data_path) {
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::ifstream in(index_path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open index: " + index_path);

    uint32_t file_npts, file_dim;
    in.read(reinterpret_cast<char*>(&file_npts),   4);
    in.read(reinterpret_cast<char*>(&file_dim),    4);
    in.read(reinterpret_cast<char*>(&start_node_), 4);

    if (file_npts != npts_ || file_dim != dim_)
        throw std::runtime_error("Index/data mismatch");

    uint32_t proj_flag = 0;
    in.read(reinterpret_cast<char*>(&proj_flag), 4);
    if (proj_flag) {
        std::string proj_path = index_path + ".proj";
        proj_.load(proj_path);
        proj_data_ = proj_.project_dataset(data_, npts_);
        proj_active_ = true;
        std::cout << "  Loaded projection: " << dim_ << "D -> "
                  << proj_.k() << "D" << std::endl;
    }

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
    std::cout << "Index loaded: " << npts_ << " pts, dim=" << dim_
              << ", working_dim=" << working_dim()
              << ", start=" << start_node_ << std::endl;
}