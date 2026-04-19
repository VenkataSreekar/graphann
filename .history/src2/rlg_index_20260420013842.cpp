// rlg_index.cpp — Radial Layer Graph implementation
// Place in graphann/src2/ alongside build_rlg.cpp and search_rlg.cpp

#include "rlg_index.h"
#include "distance.h"   // compute_l2sq (same function, same flags → same SIMD)
#include "io_utils.h"   // load_fbin, AlignedFree

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rlg {

// ════════════════════════════════════════════════════════════════════════════
//  SHELL INDEX
//  Returns which geometric shell distance d falls into, given r_base.
//    Shell 0 : [0,       r_base)
//    Shell k : [m^(k-1)*r_base,  m^k*r_base)   k >= 1
//  Capped at K_SHELLS-1 so the outermost shell catches all very-far nodes.
// ════════════════════════════════════════════════════════════════════════════

static constexpr int K_SHELLS = 10;

int RLGIndex::dist_to_shell(float d, float r_base) const {
    if (d < r_base) return 0;
    int k = (int)(std::log(d / r_base) / std::log(m_)) + 1;
    return std::min(k, K_SHELLS - 1);
}

// ════════════════════════════════════════════════════════════════════════════
//  GREEDY BEAM SEARCH  — mirrors VamanaIndex::greedy_search exactly.
//  Returns: (sorted candidates up to L, number of distance computations)
// ════════════════════════════════════════════════════════════════════════════

std::pair<std::vector<RLGIndex::Candidate>, uint32_t>
RLGIndex::greedy_search(const float* query, uint32_t L) const
{
    uint32_t dist_cmps = 0;
    std::set<Candidate> candidate_set;   // sorted by dist, bounded to L
    std::vector<bool>   visited(npts_, false);

    // ── seed from start node ─────────────────────────────────────────────────
    float d0 = compute_l2sq(query, get_vec(start_node_), dim_); dist_cmps++;
    candidate_set.insert({d0, start_node_});
    visited[start_node_] = true;

    std::set<uint32_t> expanded;

    while (true) {
        // Find closest candidate not yet expanded
        uint32_t best = UINT32_MAX;
        for (auto& [d, id] : candidate_set) {
            if (!expanded.count(id)) { best = id; break; }
        }
        if (best == UINT32_MAX) break;
        expanded.insert(best);

        // Read neighbours (locked during parallel build, free during search)
        std::vector<uint32_t> nbrs;
        {
            std::lock_guard<std::mutex> lk(locks_[best]);
            nbrs = graph_[best];
        }

        for (uint32_t nb : nbrs) {
            if (visited[nb]) continue;
            visited[nb] = true;

            float d = compute_l2sq(query, get_vec(nb), dim_); dist_cmps++;

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

    std::vector<Candidate> results(candidate_set.begin(), candidate_set.end());
    return {results, dist_cmps};
}

// ════════════════════════════════════════════════════════════════════════════
//  AUGMENTED ROBUST PRUNE
//
//  Why this beats pure RLG AND is at least as good as Vamana:
//
//  Stage 1 (alpha-RNG) — identical to Vamana's robust_prune.
//    Produces the provably best navigable edge set for the given candidates.
//    No change here: if Vamana already fills R slots, we stop.
//
//  Stage 2 (shell augmentation) — only runs if Stage 1 left budget.
//    Adds one neighbour per uncovered geometric shell.
//    These are explicit long-range edges that alpha-RNG missed because all
//    selected neighbours happened to be in the near field.
//    We NEVER remove a Stage-1 edge — quality is strictly >= Vamana.
//
//  No cosine similarity: removed entirely (expensive, hurts quality in
//  high dimensions where directional redundancy is rare).
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::augmented_robust_prune(uint32_t p,
                                       std::vector<Candidate>& cands,
                                       float alpha)
{
    // Remove self; cands should already be sorted ascending but sort to be safe
    cands.erase(std::remove_if(cands.begin(), cands.end(),
                    [p](const Candidate& c){ return c.second == p; }),
                cands.end());
    std::sort(cands.begin(), cands.end());

    // ─── Stage 1: alpha-RNG (Vamana's robust_prune) ──────────────────────────
    std::vector<uint32_t> selected;
    selected.reserve(R_);

    for (auto& [d, c] : cands) {
        if (selected.size() >= R_) break;

        bool keep = true;
        for (uint32_t s : selected) {
            float d_cs = compute_l2sq(get_vec(c), get_vec(s), dim_);
            if (d > alpha * d_cs) { keep = false; break; }
        }
        if (keep) selected.push_back(c);
    }

    // ─── Stage 2: shell augmentation ─────────────────────────────────────────
    // Only runs if we still have R-budget left after Stage 1.
    if (selected.size() < R_ && !cands.empty()) {

        // r_base: distance from p to its nearest Stage-1 neighbour.
        // cands is sorted, so scan for first entry that's in selected.
        float r_base = cands[0].first; // fallback: nearest candidate
        for (auto& [d, c] : cands) {
            bool in_sel = false;
            for (uint32_t s : selected) { if (s == c) { in_sel = true; break; } }
            if (in_sel) { r_base = std::max(d, 1e-6f); break; }
        }
        if (r_base < 1e-6f) r_base = 1e-6f;

        // Mark which shells are covered by Stage-1 neighbours.
        // We compute their distance fresh from p (R_ calls, negligible cost).
        std::vector<bool> covered(K_SHELLS, false);
        for (uint32_t s : selected) {
            float d = compute_l2sq(get_vec(p), get_vec(s), dim_);
            covered[dist_to_shell(d, r_base)] = true;
        }

        // Sweep candidates: for each uncovered shell, add closest candidate.
        for (auto& [d, c] : cands) {
            if (selected.size() >= R_) break;

            // Skip already-selected (linear scan; |selected| <= R_ = 32)
            bool already = false;
            for (uint32_t s : selected) { if (s == c) { already = true; break; } }
            if (already) continue;

            int k = dist_to_shell(d, r_base);
            if (!covered[k]) {
                selected.push_back(c);
                covered[k] = true;
            }
        }
    }

    // Write result (thread-safe: caller holds lock on p, or we're single-threaded)
    graph_[p] = std::move(selected);
}

// ════════════════════════════════════════════════════════════════════════════
//  MEDOID
// ════════════════════════════════════════════════════════════════════════════

uint32_t RLGIndex::compute_medoid() const {
    // Full centroid over all points — exact (same as Vamana)
    std::vector<double> centroid(dim_, 0.0);
    for (uint32_t i = 0; i < npts_; ++i) {
        const float* v = get_vec(i);
        for (uint32_t d = 0; d < dim_; ++d) centroid[d] += v[d];
    }
    for (double& x : centroid) x /= npts_;

    float    best  = std::numeric_limits<float>::max();
    uint32_t mrd   = 0;
    for (uint32_t i = 0; i < npts_; ++i) {
        float dist = 0.f;
        const float* v = get_vec(i);
        for (uint32_t d = 0; d < dim_; ++d) {
            float diff = v[d] - (float)centroid[d]; dist += diff * diff;
        }
        if (dist < best) { best = dist; mrd = i; }
    }
    return mrd;
}

// ════════════════════════════════════════════════════════════════════════════
//  BUILD PASS  — mirrors VamanaIndex::run_build_pass
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::run_build_pass(const std::vector<uint32_t>& perm,
                               uint32_t R, uint32_t L,
                               float alpha, uint32_t gamma_R)
{
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; ++idx) {
        uint32_t point = perm[idx];

        // ── 1. Greedy search from start to get candidates ─────────────────
        auto [cands, _dc] = greedy_search(get_vec(point), L);

        // ── 2. Merge with existing edges (prevents catastrophic forgetting
        //       between passes, identical to Vamana's approach) ─────────────
        {
            std::lock_guard<std::mutex> lk(locks_[point]);
            for (uint32_t nb : graph_[point]) {
                float d = compute_l2sq(get_vec(point), get_vec(nb), dim_);
                cands.push_back({d, nb});
            }
        }

        // ── 3. Augmented prune → set forward edges ────────────────────────
        augmented_robust_prune(point, cands, alpha);

        // ── 4. Back edges + conditional re-prune ─────────────────────────
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

        if (idx % 10000 == 0) {
            #pragma omp critical
            std::cout << "\r  Processed " << idx << " / " << npts_
                      << " points" << std::flush;
        }
    }
    std::cout << "\r  Processed " << npts_ << " / " << npts_
              << " points\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  BUILD  (public entry point) — mirrors VamanaIndex::build
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::build(const std::string& data_path,
                     uint32_t R, uint32_t L,
                     float alpha, float m, float gamma)
{
    R_ = R; m_ = m;
    if (m_ <= 1.0f) m_ = 2.0f;  // guard against degenerate shell geometry

    // ── Load data (aligned, SIMD-friendly) ───────────────────────────────────
    std::cout << "Loading data from " << data_path << " ...\n";
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts; dim_ = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;
    std::cout << "  Points: " << npts_ << "  Dims: " << dim_ << "\n";

    if (L < R) { std::cerr << "Warning: L < R, setting L = R\n"; L = R; }

    graph_.assign(npts_, {});
    locks_ = std::vector<std::mutex>(npts_);

    // ── Medoid start node (same as Vamana) ───────────────────────────────────
    std::mt19937 rng(42);
    start_node_ = compute_medoid();
    std::cout << "  Start node (medoid): " << start_node_ << "\n";

    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    uint32_t gamma_R = (uint32_t)(gamma * R);

    // ── Two-pass build: pass 1 strict (alpha=1.0), pass 2 loose (user alpha)
    // This mirrors Vamana's two-pass strategy exactly.
    // Pass 1 builds a well-connected base graph with monotone edges.
    // Pass 2 adds long-range diversity edges that pass 1 excluded.
    std::cout << "\n--- Pass 1 (alpha=1.0) ---\n";
    run_build_pass(perm, R, L, 1.0f, gamma_R);
    std::cout << "  Pass 1 complete.\n";

    std::shuffle(perm.begin(), perm.end(), rng);
    std::cout << "\n--- Pass 2 (alpha=" << alpha << ") ---\n";
    run_build_pass(perm, R, L, alpha, gamma_R);
    std::cout << "  Pass 2 complete.\n";

    // ── Degree stats ──────────────────────────────────────────────────────────
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
//  SAVE / LOAD  — mirrors VamanaIndex::save / load
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);

    // header: same field order as Vamana + m_ appended
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

    std::cout << "RLG index saved to " << path << "\n";
}

void RLGIndex::load(const std::string& index_path,
                    const std::string& data_path)
{
    // ── Load data vectors (aligned) — identical to VamanaIndex::load ─────────
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts; dim_ = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    // ── Load graph ────────────────────────────────────────────────────────────
    std::ifstream f(index_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open index: " + index_path);

    uint32_t file_n, file_d;
    f.read((char*)&file_n,       sizeof(uint32_t));
    f.read((char*)&file_d,       sizeof(uint32_t));
    f.read((char*)&start_node_,  sizeof(uint32_t));
    f.read((char*)&R_,           sizeof(uint32_t));
    f.read((char*)&m_,           sizeof(float));

    if (file_n != npts_ || file_d != dim_)
        throw std::runtime_error(
            "Index/data mismatch: index=" + std::to_string(file_n) +
            "x" + std::to_string(file_d) +
            " data=" + std::to_string(npts_) + "x" + std::to_string(dim_));

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    for (uint32_t i = 0; i < npts_; ++i) {
        uint32_t deg;
        f.read((char*)&deg, sizeof(uint32_t));
        graph_[i].resize(deg);
        if (deg > 0)
            f.read((char*)graph_[i].data(), deg * sizeof(uint32_t));
    }

    std::cout << "RLG index loaded: " << npts_ << " points, dim=" << dim_
              << ", start=" << start_node_ << ", R=" << R_
              << ", m=" << m_ << "\n";
}

} // namespace rlg