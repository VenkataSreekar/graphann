// rlg_index.cpp  — Radial Layer Graph implementation
// Drop this file into graphann/src/ alongside vamana_index.cpp

#include "rlg_index.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rlg {

// ════════════════════════════════════════════════════════════════════════════
//  GREEDY BEAM SEARCH  (identical logic to VamanaIndex::search internals)
// ════════════════════════════════════════════════════════════════════════════

std::vector<Candidate> RLGIndex::greedy_search(
    const float* q, int start, int L) const
{
    std::vector<bool> visited(n_, false);
    std::set<Candidate> frontier;   // candidates still to expand
    std::set<Candidate> results;    // best L seen so far

    float d0 = l2sq(q, node(start), dim_);
    Candidate c0{d0, start};
    frontier.insert(c0);
    results.insert(c0);
    visited[start] = true;

    while (!frontier.empty()) {
        Candidate cur = *frontier.begin();
        frontier.erase(frontier.begin());

        // prune: if cur is worse than our worst result and results is full → stop
        if ((int)results.size() >= L &&
            cur.dist > std::prev(results.end())->dist)
            break;

        for (int nb : adj_[cur.id]) {
            if (visited[nb]) continue;
            visited[nb] = true;

            float d = l2sq(q, node(nb), dim_);
            Candidate cnb{d, nb};

            if ((int)results.size() < L) {
                results.insert(cnb);
                frontier.insert(cnb);
            } else if (d < std::prev(results.end())->dist) {
                results.erase(std::prev(results.end()));
                results.insert(cnb);
                frontier.insert(cnb);
            }
        }
    }

    return {results.begin(), results.end()};
}

// ════════════════════════════════════════════════════════════════════════════
//  RADIAL PRUNE  — the key new algorithm replacing robust_prune
//
//  Given a sorted candidate list for node p:
//   1.  r_base = dist(p, closest candidate)   ← anchors to local density
//   2.  Bin each candidate into shell k = floor(log_m(dist/r_base)) + 1
//       (shell 0 for dist < r_base)
//   3.  Distribute R slots across shells with harmonic decay
//       budget[k] = floor(R / (k+1) / H_K)  where H_K = Σ 1/(k+1)
//   4.  Within each shell, greedily pick candidates that are angularly
//       diverse: cos_sim(p→c, p→already_selected) < alpha_ang
// ════════════════════════════════════════════════════════════════════════════

std::vector<int> RLGIndex::radial_prune(
    int p, std::vector<Candidate>& cands) const
{
    if (cands.empty()) return {};

    const float* p_data = node(p);
    const int    K      = 12;        // max number of shells

    // ── 1. r_base ────────────────────────────────────────────────────────────
    float r_base = cands[0].dist;
    if (r_base < 1e-12f) r_base = 1e-6f;

    // ── 2. bin into shells ───────────────────────────────────────────────────
    std::vector<std::vector<Candidate>> shells(K);
    for (auto& c : cands) {
        if (c.id == p) continue;
        int k;
        if (c.dist < r_base) {
            k = 0;
        } else {
            k = (int)std::floor(std::log(c.dist / r_base) / std::log(m_)) + 1;
            k = std::min(k, K - 1);
        }
        shells[k].push_back(c);
    }

    // ── 3. harmonic slot budget ──────────────────────────────────────────────
    // Count non-empty shells so we don't waste budget
    int non_empty = 0;
    for (int k = 0; k < K; ++k) if (!shells[k].empty()) non_empty++;
    if (non_empty == 0) return {};

    // H_K = sum of 1/(k+1) over non-empty shells (for normalisation)
    // We iterate over all K shells; empty shells get budget 0 anyway.
    float H = 0.f;
    for (int k = 0; k < K; ++k)
        if (!shells[k].empty()) H += 1.f / (k + 1.f);

    std::vector<int> budget(K, 0);
    int assigned = 0;
    for (int k = 0; k < K; ++k) {
        if (shells[k].empty()) continue;
        budget[k] = std::max(1, (int)std::round((float)R_ / ((k + 1.f) * H)));
        assigned += budget[k];
    }
    // Clamp total to R_
    // If over: shave from the most-loaded inner shell
    // If under: add to shell 0
    while (assigned > R_) {
        for (int k = 0; k < K && assigned > R_; ++k) {
            if (budget[k] > 1) { budget[k]--; assigned--; }
        }
    }
    while (assigned < R_) { budget[0]++; assigned++; }

    // ── 4. angular-diversity selection per shell ─────────────────────────────
    std::vector<int> selected;
    selected.reserve(R_);

    for (int k = 0; k < K && (int)selected.size() < R_; ++k) {
        if (shells[k].empty()) continue;
        int b = std::min(budget[k], R_ - (int)selected.size());

        // Build direction vectors from p to each candidate in this shell
        std::vector<std::vector<float>> dirs;
        dirs.reserve(shells[k].size());
        for (auto& c : shells[k]) {
            const float* cd = node(c.id);
            std::vector<float> dir(dim_);
            for (int d = 0; d < dim_; ++d) dir[d] = cd[d] - p_data[d];
            dirs.push_back(std::move(dir));
        }

        // Greedy: pick the closest candidate, then skip any too-similar ones
        std::vector<int> picked_in_shell;
        for (int i = 0; i < (int)shells[k].size() && (int)picked_in_shell.size() < b; ++i) {
            bool diverse = true;
            for (int j : picked_in_shell) {
                if (cos_sim(dirs[i].data(), dirs[j].data(), dim_) > alpha_ang_) {
                    diverse = false;
                    break;
                }
            }
            if (diverse) {
                picked_in_shell.push_back(i);
                selected.push_back(shells[k][i].id);
            }
        }

        // Fallback: if angular filter was too strict, just take the closest
        if (picked_in_shell.empty()) {
            selected.push_back(shells[k][0].id);
        }
    }

    return selected;
}

// ════════════════════════════════════════════════════════════════════════════
//  MEDOID  — point closest to the dataset centroid (better start than random)
// ════════════════════════════════════════════════════════════════════════════

int RLGIndex::medoid(int sample) const {
    sample = std::min(sample, n_);
    std::mt19937 rng(42);
    std::vector<int> idx(n_);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(sample);

    std::vector<float> centroid(dim_, 0.f);
    for (int i : idx) {
        const float* p = node(i);
        for (int d = 0; d < dim_; ++d) centroid[d] += p[d];
    }
    for (float& x : centroid) x /= sample;

    float best = std::numeric_limits<float>::max();
    int   best_id = 0;
    for (int i : idx) {
        float d = l2sq(centroid.data(), node(i), dim_);
        if (d < best) { best = d; best_id = i; }
    }
    return best_id;
}

// ════════════════════════════════════════════════════════════════════════════
//  INTRINSIC DIMENSION (Levina-Bickel 2-NN estimator)
// ════════════════════════════════════════════════════════════════════════════

float RLGIndex::intrinsic_dim(int sample) const {
    sample = std::min(sample, n_);
    std::mt19937 rng(42);
    std::vector<int> idx(n_);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(sample);

    double sum = 0.0; int cnt = 0;
    for (int i : idx) {
        float d1 = std::numeric_limits<float>::max();
        float d2 = std::numeric_limits<float>::max();
        for (int j : idx) {
            if (j == i) continue;
            float d = l2sq(node(i), node(j), dim_);
            if (d < d1)      { d2 = d1; d1 = d; }
            else if (d < d2) { d2 = d; }
        }
        if (d1 > 1e-12f && d2 > d1) {
            sum += std::log(d2 / d1); cnt++;
        }
    }
    if (cnt == 0 || sum < 1e-9) return (float)dim_;
    return (float)cnt / (float)sum;   // = 1 / E[log(d2/d1)]
}

// ════════════════════════════════════════════════════════════════════════════
//  BUILD PASS  (one full pass over all nodes, parallel)
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::build_pass(int L_build) {
    std::vector<int> order(n_);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(std::random_device{}());
    std::shuffle(order.begin(), order.end(), rng);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int idx = 0; idx < n_; ++idx) {
        int p = order[idx];

        // ── greedy search to get candidates ──────────────────────────────────
        auto cands = greedy_search(node(p), start_node_, L_build);

        // remove self
        cands.erase(
            std::remove_if(cands.begin(), cands.end(),
                           [p](const Candidate& c){ return c.id == p; }),
            cands.end());

        // ── radial prune → forward edges ─────────────────────────────────────
        auto neighbors = radial_prune(p, cands);

        {
#ifdef _OPENMP
            std::lock_guard<std::mutex> lk(mtx_[p]);
#endif
            adj_[p] = neighbors;
        }

        // ── backward edges (navigability guarantee) ───────────────────────────
        // ── backward edges (navigability guarantee) ───────────────────────────
        for (int nb : neighbors) {
#ifdef _OPENMP
            std::lock_guard<std::mutex> lk(mtx_[nb]);
#endif
            auto& nb_adj = adj_[nb];
            if (std::find(nb_adj.begin(), nb_adj.end(), p) == nb_adj.end()) {
                nb_adj.push_back(p);
                // OPTIMIZATION: Only re-prune if the list gets 20% too large.
                // This stops threads from getting stuck waiting on the lock!
                if ((int)nb_adj.size() > (int)(R_ * 1.2)) { {
                    // re-prune nb with all its current neighbors as candidates
                    std::vector<Candidate> nb_cands;
                    nb_cands.reserve(nb_adj.size());
                    const float* nb_data = node(nb);
                    for (int x : nb_adj) {
                        nb_cands.push_back({l2sq(nb_data, node(x), dim_), x});
                    }
                    std::sort(nb_cands.begin(), nb_cands.end());
                    nb_adj = radial_prune(nb, nb_cands);
                }
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  BUILD  (public entry point)
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::build(const float* data, int n, int dim,
                     int R, float m, int L_build,
                     float alpha_ang, bool two_pass)
{
    n_         = n;
    dim_       = dim;
    R_         = R;
    m_         = m;
    alpha_ang_ = alpha_ang;

    // copy data
    data_.assign(data, data + (size_t)n * dim);

    adj_.assign(n, {});
    mtx_ = std::vector<std::mutex>(n);

    // ── auto-select m if caller passed m <= 0 ────────────────────────────────
    if (m_ <= 0.f) {
        float id = intrinsic_dim(1000);
        m_ = std::pow(2.f, 1.f / std::max(1.f, id));
        std::cout << "[RLG] intrinsic_dim ≈ " << id
                  << "  →  auto m = " << m_ << "\n";
    }

    // ── medoid start node ────────────────────────────────────────────────────
    start_node_ = medoid(2000);
    std::cout << "[RLG] start_node (medoid) = " << start_node_ << "\n";

    // ── bootstrap: random single neighbor per node ───────────────────────────
    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, n - 1);
        for (int i = 0; i < n; ++i) {
            int nb = dist(rng);
            if (nb != i) adj_[i].push_back(nb);
        }
    }

    std::cout << "[RLG] Pass 1 ...\n";
    build_pass(L_build);

    if (two_pass) {
        std::cout << "[RLG] Pass 2 (refinement) ...\n";
        build_pass(L_build);
    }

    std::cout << "[RLG] Build done.  n=" << n << "  dim=" << dim
              << "  R=" << R << "  m=" << m_ << "\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  SEARCH  (public)
// ════════════════════════════════════════════════════════════════════════════

std::vector<int> RLGIndex::search(const float* query, int K, int L_search) const {
    auto cands = greedy_search(query, start_node_, L_search);
    std::vector<int> result;
    result.reserve(K);
    for (int i = 0; i < std::min(K, (int)cands.size()); ++i)
        result.push_back(cands[i].id);
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
//  DEGREE STATS
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::degree_stats(float& mean, float& stddev, float& gini) const {
    std::vector<float> degs(n_);
    for (int i = 0; i < n_; ++i) degs[i] = (float)adj_[i].size();

    float s = 0.f;
    for (float d : degs) s += d;
    mean = s / n_;

    float ss = 0.f;
    for (float d : degs) ss += (d - mean) * (d - mean);
    stddev = std::sqrt(ss / n_);

    std::sort(degs.begin(), degs.end());
    float si = 0.f;
    for (int i = 0; i < n_; ++i) si += (float)(i + 1) * degs[i];
    gini = (2.f * si) / ((float)n_ * s) - (float)(n_ + 1) / (float)n_;
}

// ════════════════════════════════════════════════════════════════════════════
//  SAVE / LOAD
// ════════════════════════════════════════════════════════════════════════════

void RLGIndex::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);

    // header
    f.write((char*)&n_,         sizeof(int));
    f.write((char*)&dim_,       sizeof(int));
    f.write((char*)&R_,         sizeof(int));
    f.write((char*)&m_,         sizeof(float));
    f.write((char*)&alpha_ang_, sizeof(float));
    f.write((char*)&start_node_,sizeof(int));

    // adjacency
    for (int i = 0; i < n_; ++i) {
        int deg = (int)adj_[i].size();
        f.write((char*)&deg, sizeof(int));
        if (deg > 0)
            f.write((char*)adj_[i].data(), deg * sizeof(int));
    }
}

void RLGIndex::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for reading: " + path);

    f.read((char*)&n_,         sizeof(int));
    f.read((char*)&dim_,       sizeof(int));
    f.read((char*)&R_,         sizeof(int));
    f.read((char*)&m_,         sizeof(float));
    f.read((char*)&alpha_ang_, sizeof(float));
    f.read((char*)&start_node_,sizeof(int));

    adj_.resize(n_);
    for (int i = 0; i < n_; ++i) {
        int deg;
        f.read((char*)&deg, sizeof(int));
        adj_[i].resize(deg);
        if (deg > 0)
            f.read((char*)adj_[i].data(), deg * sizeof(int));
    }
}

} // namespace rlg
