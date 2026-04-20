// rlg_index.cpp — Radial Layer Graph
// Place in graphann/src2/

#include "rlg_index.h"
#include "distance.h"
#include "io_utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rlg {

static constexpr int K_SHELLS = 10;  // max geometric shells per node

// ════════════════════════════════════════════════════════════════════════════
//  SHELL INDEX
// ════════════════════════════════════════════════════════════════════════════

int RLGIndex::dist_to_shell(float d, float r_base) const {
    if (d < r_base) return 0;
    int k = (int)(std::log(d / r_base) / std::log(m_)) + 1;
    return std::min(k, K_SHELLS - 1);
}

// ════════════════════════════════════════════════════════════════════════════
//  GREEDY BEAM SEARCH  —  returns (sorted_results, dist_cmps)
//  Identical to VamanaIndex::greedy_search.  NO locks during search
//  (graph is read-only during query time).
// ════════════════════════════════════════════════════════════════════════════

std::pair<std::vector<RLGIndex::Candidate>, uint32_t>
RLGIndex::greedy_search(const float* query, uint32_t L) const
{
    // --- SAFE PERSISTENT SCRATCH BUFFER ---
    // Using raw pointers avoids the thread_local destructor crash on Windows.
    static thread_local uint32_t* visited_tags = nullptr;
    static thread_local uint32_t tags_size = 0;
    static thread_local uint32_t current_query_id = 0;

    // Allocate or resize only when necessary
    if (visited_tags == nullptr || tags_size != npts_) {
        if (visited_tags) std::free(visited_tags);
        // Use calloc to initialize with zeros
        visited_tags = (uint32_t*)std::calloc(npts_, sizeof(uint32_t));
        tags_size = npts_;
        current_query_id = 0;
    }

    current_query_id++;
    // Handle 32-bit wrap-around (after 4 billion queries)
    if (current_query_id == 0) {
        memset(visited_tags, 0, npts_ * sizeof(uint32_t));
        current_query_id = 1;
    }

    auto is_visited = [&](uint32_t id) { return visited_tags[id] == current_query_id; };
    auto set_visited = [&](uint32_t id) { visited_tags[id] = current_query_id; };
    // --------------------------------------

    uint32_t dist_cmps = 0;
    std::set<Candidate> candidate_set;

    float d0 = compute_l2sq(query, get_vec(start_node_), dim_);
    dist_cmps++;
    candidate_set.insert({d0, start_node_});
    set_visited(start_node_);

    std::set<uint32_t> expanded;

    while (true) {
        uint32_t best = UINT32_MAX;
        for (auto& [d, id] : candidate_set)
            if (!expanded.count(id)) { best = id; break; }
        if (best == UINT32_MAX) break;
        expanded.insert(best);

        // During build, we hold a lock when writing graph_[node].
        // During search, graph is read-only — no lock needed.
        // We copy into a local vector to keep the critical section short
        // even during the build phase.
        std::vector<uint32_t> nbrs;
        {
            std::lock_guard<std::mutex> lk(locks_[best]);
            nbrs = graph_[best];
        }

        for (uint32_t nb : nbrs) {
            if (is_visited(nb)) continue;
            set_visited(nb);

            float d = compute_l2sq(query, get_vec(nb), dim_);
            dist_cmps++;

            if (candidate_set.size() < L) {
                candidate_set.insert({d, nb});
            } else {
                auto worst = std::prev(candidate_set.end());
                if (d < worst->first) {
                    candidate_set.erase(worst);
                    candidate_set.insert({d, nb});
                }
            }
        }
    }

    return {{candidate_set.begin(), candidate_set.end()}, dist_cmps};
}

// ════════════════════════════════════════════════════════════════════════════
//  AUGMENTED ROBUST PRUNE
//
//  The neighbour budget R_ is split into two parts:
//
//    SHELL_SLOTS = R_ / 4  (e.g. 8 out of 32)
//    RNG_LIMIT   = R_ - SHELL_SLOTS  (e.g. 24 out of 32)
//
//  Stage 1: Standard alpha-RNG fills up to RNG_LIMIT neighbours.
//           These provide the same local navigation quality as Vamana.
//           (alpha same meaning as Vamana: 1.0–1.5)
//
//  Stage 2: Fills remaining SHELL_SLOTS with the nearest candidate from
//           each unrepresented geometric distance shell.
//           This ALWAYS runs (not conditional on Stage 1 leaving budget).
//           These provide explicit multi-scale global coverage (Kleinberg 2000).
//
//  Result:  At least as good as Vamana locally, plus guaranteed scale diversity.
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::augmented_robust_prune(uint32_t node,
                                       std::vector<Candidate>& cands,
                                       float alpha)
{
    // Remove self, sort ascending
    cands.erase(std::remove_if(cands.begin(), cands.end(),
                    [node](const Candidate& c){ return c.second == node; }),
                cands.end());
    std::sort(cands.begin(), cands.end());
    if (cands.empty()) { graph_[node].clear(); return; }

    // Budget split
    const int SHELL_SLOTS = std::max(1, (int)(R_ / 4));
    const int RNG_LIMIT   = (int)R_ - SHELL_SLOTS;

    // ── Stage 1: alpha-RNG (Vamana logic, capped at RNG_LIMIT) ──────────────
    std::vector<uint32_t> selected;
    selected.reserve(R_);

    for (auto& [d, c] : cands) {
        if ((int)selected.size() >= RNG_LIMIT) break;
        bool keep = true;
        for (uint32_t s : selected) {
            float dcs = compute_l2sq(get_vec(c), get_vec(s), dim_);
            if (d > alpha * dcs) { keep = false; break; }
        }
        if (keep) selected.push_back(c);
    }

    // ── Stage 2: shell augmentation (ALWAYS runs) ────────────────────────────
    // Anchor shells on distance to the nearest Stage-1 neighbour.
    float r_base = cands[0].first;  // fallback: nearest candidate overall
    if (!selected.empty()) {
        // Use nearest SELECTED neighbour as r_base
        float min_d = std::numeric_limits<float>::max();
        for (uint32_t s : selected) {
            float d = compute_l2sq(get_vec(node), get_vec(s), dim_);
            if (d < min_d) min_d = d;
        }
        r_base = std::max(min_d, 1e-6f);
    }

    // Which shells are already covered by Stage-1 neighbours?
    std::vector<bool> covered(K_SHELLS, false);
    for (uint32_t s : selected) {
        float d = compute_l2sq(get_vec(node), get_vec(s), dim_);
        covered[dist_to_shell(d, r_base)] = true;
    }

    // Build a fast lookup for already-selected IDs
    // (R_ <= 32, so linear scan over selected is O(32) — perfectly fine)
    auto already_selected = [&](uint32_t id) {
        for (uint32_t s : selected) if (s == id) return true;
        return false;
    };

    // Sweep sorted candidates: for each uncovered shell, take the nearest one
    int shell_added = 0;
    for (auto& [d, c] : cands) {
        if ((int)selected.size() >= (int)R_) break;
        if (shell_added >= SHELL_SLOTS) break;
        if (already_selected(c)) continue;

        int k = dist_to_shell(d, r_base);
        if (!covered[k]) {
            selected.push_back(c);
            covered[k] = true;
            shell_added++;
        }
    }

    // ── Fill any remaining slots with plain alpha-RNG (no shell required) ────
    for (auto& [d, c] : cands) {
        if ((int)selected.size() >= (int)R_) break;
        if (already_selected(c)) continue;
        bool keep = true;
        for (uint32_t s : selected) {
            float dcs = compute_l2sq(get_vec(c), get_vec(s), dim_);
            if (d > alpha * dcs) { keep = false; break; }
        }
        if (keep) selected.push_back(c);
    }

    // Write result (caller holds locks_[node] or we're in single-threaded path)
    {
        std::lock_guard<std::mutex> lk(locks_[node]);
        graph_[node] = std::move(selected);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  MEDOID  (same as Vamana)
// ════════════════════════════════════════════════════════════════════════════

uint32_t RLGIndex::compute_medoid() const {
    std::vector<double> centroid(dim_, 0.0);
    for (uint32_t i = 0; i < npts_; ++i) {
        const float* v = get_vec(i);
        for (uint32_t d = 0; d < dim_; ++d) centroid[d] += v[d];
    }
    for (double& x : centroid) x /= npts_;

    float best = std::numeric_limits<float>::max();
    uint32_t med = 0;
    for (uint32_t i = 0; i < npts_; ++i) {
        float dist = 0.f;
        const float* v = get_vec(i);
        for (uint32_t d = 0; d < dim_; ++d) {
            float diff = v[d] - (float)centroid[d]; dist += diff * diff;
        }
        if (dist < best) { best = dist; med = i; }
    }
    return med;
}

// ════════════════════════════════════════════════════════════════════════════
//  BUILD PASS  (same structure as VamanaIndex::run_build_pass)
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::run_build_pass(const std::vector<uint32_t>& perm,
                               uint32_t R, uint32_t L,
                               float alpha, uint32_t gamma_R)
{
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; ++idx) {
        uint32_t point = perm[idx];

        // 1. Greedy search for candidates
        auto [cands, _dc] = greedy_search(get_vec(point), L);

        // 2. Neighbourhood merging (prevents catastrophic forgetting between passes)
        {
            std::lock_guard<std::mutex> lk(locks_[point]);
            for (uint32_t nb : graph_[point]) {
                float d = compute_l2sq(get_vec(point), get_vec(nb), dim_);
                cands.push_back({d, nb});
            }
        }

        // 3. Augmented prune → set forward edges
        augmented_robust_prune(point, cands, alpha);

        // 4. Back edges + conditional re-prune
        std::vector<uint32_t> my_nbrs;
        {
            std::lock_guard<std::mutex> lk(locks_[point]);
            my_nbrs = graph_[point];
        }

        for (uint32_t nb : my_nbrs) {
            bool needs_prune = false;
            {
                std::lock_guard<std::mutex> lk(locks_[nb]);
                graph_[nb].push_back(point);
                needs_prune = (graph_[nb].size() > gamma_R);
            }
            if (needs_prune) {
                std::vector<Candidate> nb_cands;
                {
                    std::lock_guard<std::mutex> lk(locks_[nb]);
                    nb_cands.reserve(graph_[nb].size());
                    for (uint32_t x : graph_[nb]) {
                        float d = compute_l2sq(get_vec(nb), get_vec(x), dim_);
                        nb_cands.push_back({d, x});
                    }
                }
                augmented_robust_prune(nb, nb_cands, alpha);
            }
        }

        if (idx % 50000 == 0) {
            #pragma omp critical
            std::cout << "\r  Processed " << idx << " / " << npts_
                      << " points" << std::flush;
        }
    }
    std::cout << "\r  Processed " << npts_ << " / " << npts_
              << " points\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  BUILD  (two-pass, same structure as Vamana)
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::build(const std::string& data_path,
                     uint32_t R, uint32_t L,
                     float alpha, float m, float gamma)
{
    R_ = R;
    m_ = (m <= 1.0f) ? 2.0f : m;  // guard against degenerate shell geometry

    std::cout << "Loading data from " << data_path << "...\n";
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts; dim_ = mat.dims;
    data_ = mat.data.release(); owns_data_ = true;
    std::cout << "  Points: " << npts_ << "  Dims: " << dim_ << "\n";

    if (L < R) { std::cerr << "Warning: L < R, setting L = R\n"; L = R; }

    graph_.assign(npts_, {});
    locks_ = std::vector<std::mutex>(npts_);

    std::mt19937 rng(42);
    start_node_ = compute_medoid();
    std::cout << "  Start node (medoid): " << start_node_ << "\n";

    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    uint32_t gamma_R = (uint32_t)(gamma * R);

    // Two-pass build: same structure as Vamana.
    // Pass 1 (alpha=1.0): strict RNG builds a well-connected monotone base graph.
    // Pass 2 (alpha=user): relaxes RNG to add long-range diversity edges.
    // Shell augmentation activates in BOTH passes.
    std::cout << "\n--- Pass 1 (alpha=1.0, shell augmentation ON) ---\n";
    run_build_pass(perm, R, L, 1.0f, gamma_R);
    std::cout << "  Pass 1 complete.\n";

    std::shuffle(perm.begin(), perm.end(), rng);
    std::cout << "\n--- Pass 2 (alpha=" << alpha << ", shell augmentation ON) ---\n";
    run_build_pass(perm, R, L, alpha, gamma_R);
    std::cout << "  Pass 2 complete.\n";

    size_t total = 0;
    for (uint32_t i = 0; i < npts_; ++i) total += graph_[i].size();
    std::cout << "\n  Average out-degree: " << (double)total / npts_ << "\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  SEARCH
// ════════════════════════════════════════════════════════════════════════════

SearchResult RLGIndex::search(const float* query, uint32_t K, uint32_t L) const {
    if (L < K) L = K;
    Timer t;
    auto [cands, dist_cmps] = greedy_search(query, L);
    double latency = t.elapsed_us();

    SearchResult res;
    res.dist_cmps  = dist_cmps;
    res.latency_us = latency;
    res.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < cands.size(); ++i)
        res.ids.push_back(cands[i].second);
    return res;
}

// ════════════════════════════════════════════════════════════════════════════
//  DEGREE STATS
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::degree_stats(float& mean, float& stddev, float& gini) const {
    std::vector<float> degs(npts_);
    for (uint32_t i = 0; i < npts_; ++i) degs[i] = (float)graph_[i].size();

    float s = 0.f;
    for (float d : degs) s += d;
    mean = s / npts_;

    float ss = 0.f;
    for (float d : degs) ss += (d - mean) * (d - mean);
    stddev = std::sqrt(ss / npts_);

    std::sort(degs.begin(), degs.end());
    float si = 0.f;
    for (uint32_t i = 0; i < npts_; ++i) si += (float)(i + 1) * degs[i];
    gini = (2.f * si) / ((float)npts_ * s) - (float)(npts_ + 1) / (float)npts_;
}

// ════════════════════════════════════════════════════════════════════════════
//  SAVE / LOAD
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);

    f.write((char*)&npts_,       sizeof(uint32_t));
    f.write((char*)&dim_,        sizeof(uint32_t));
    f.write((char*)&start_node_, sizeof(uint32_t));
    f.write((char*)&R_,          sizeof(uint32_t));
    f.write((char*)&m_,          sizeof(float));

    for (uint32_t i = 0; i < npts_; ++i) {
        uint32_t deg = (uint32_t)graph_[i].size();
        f.write((char*)&deg, sizeof(uint32_t));
        if (deg > 0)
            f.write((char*)graph_[i].data(), deg * sizeof(uint32_t));
    }
    std::cout << "Index saved to " << path << "\n";
}

void RLGIndex::load(const std::string& index_path,
                    const std::string& data_path)
{
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts; dim_ = mat.dims;
    data_ = mat.data.release(); owns_data_ = true;

    std::ifstream f(index_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + index_path);

    uint32_t fn, fd;
    f.read((char*)&fn,         sizeof(uint32_t));
    f.read((char*)&fd,         sizeof(uint32_t));
    f.read((char*)&start_node_,sizeof(uint32_t));
    f.read((char*)&R_,         sizeof(uint32_t));
    f.read((char*)&m_,         sizeof(float));

    if (fn != npts_ || fd != dim_)
        throw std::runtime_error("Index/data mismatch");

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);
    for (uint32_t i = 0; i < npts_; ++i) {
        uint32_t deg; f.read((char*)&deg, sizeof(uint32_t));
        graph_[i].resize(deg);
        if (deg > 0)
            f.read((char*)graph_[i].data(), deg * sizeof(uint32_t));
    }
    std::cout << "Index loaded: " << npts_ << " points, dim=" << dim_
              << ", start=" << start_node_
              << ", R=" << R_ << ", m=" << m_ << "\n";
}

} // namespace rlg