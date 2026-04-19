#include "radial_layer_index.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rlg {

// =============================================================
//  SHELL GEOMETRY
// =============================================================

ShellInfo RadialLayerIndex::compute_shells(float r_base) const {
    ShellInfo si;
    si.r_base = r_base;
    si.m      = m_;
    // K = number of shells needed so that m^K * r_base covers "infinity"
    // In practice, we cap at K=12; the outermost shell catches everything beyond.
    si.K = 12;
    return si;
}

int RadialLayerIndex::shell_index(const ShellInfo& si, float dist) const {
    // Shell k covers [m^(k-1)*r_base, m^k*r_base)
    // Shell 0 covers [0, r_base)
    if (dist < si.r_base) return 0;
    float ratio = dist / si.r_base;
    int k = (int)std::floor(std::log(ratio) / std::log(si.m)) + 1;
    return std::min(k, si.K - 1); // clamp to last shell
}

// Allocate R total budget across shells.
// Strategy: geometric decay — inner shells get more slots.
// shell_quota[k] = floor(R * decay^k / sum)  where decay < 1
std::vector<int> RadialLayerIndex::shell_budget(
    const ShellInfo& si, int R_total, int non_empty_shells) const
{
    if (non_empty_shells <= 0) non_empty_shells = 1;
    int K = si.K;
    std::vector<int> quota(K, 0);

    // Weight for shell k: w_k = 1 / (k+1)  — harmonic decay
    // Gives more budget to inner shells, but outer shells always get >= 1
    std::vector<float> w(K, 0.f);
    float total_w = 0.f;
    for (int k = 0; k < K; ++k) {
        w[k] = 1.f / (k + 1.f);
        total_w += w[k];
    }

    int assigned = 0;
    for (int k = 0; k < K; ++k) {
        quota[k] = std::max(1, (int)std::round(R_total * w[k] / total_w));
        assigned += quota[k];
    }

    // Adjust to exactly R_total
    // Add/remove from innermost shell
    while (assigned > R_total) { quota[0]--; assigned--; }
    while (assigned < R_total) { quota[0]++; assigned++; }

    return quota;
}

// =============================================================
//  RADIAL PRUNE  (the key new algorithm)
// =============================================================
//
//  Given sorted candidates for node p:
//  1. Compute r_base = dist to nearest candidate (1-NN)
//  2. Compute shell boundaries: [m^k * r_base, m^(k+1) * r_base)
//  3. Compute per-shell budget
//  4. For each shell, greedily pick candidates with angular diversity:
//     when adding candidate c to shell k's selected set S_k,
//     ensure cos_sim(p->c, p->s) < alpha for all s in S_k
//
std::vector<int> RadialLayerIndex::radial_prune(
    int p, const std::vector<Candidate>& candidates) const
{
    if (candidates.empty()) return {};

    const float* p_data = node_data(p);

    // Step 1: r_base = nearest neighbor distance
    float r_base = candidates[0].dist;
    if (r_base < 1e-12f) r_base = 1e-6f;  // degenerate guard

    ShellInfo si = compute_shells(r_base);

    // Step 2: Group candidates by shell
    int K = si.K;
    std::vector<std::vector<Candidate>> shells(K);
    for (auto& cand : candidates) {
        if (cand.id == p) continue;
        int k = shell_index(si, cand.dist);
        shells[k].push_back(cand);
    }

    // Count non-empty shells
    int non_empty = 0;
    for (int k = 0; k < K; ++k)
        if (!shells[k].empty()) non_empty++;

    // Step 3: Compute per-shell budget
    auto quota = shell_budget(si, R_, non_empty);

    // Step 4: Select from each shell with angular diversity
    std::vector<int> selected;
    selected.reserve(R_);

    for (int k = 0; k < K && (int)selected.size() < R_; ++k) {
        if (shells[k].empty()) continue;

        int budget = std::min(quota[k], R_ - (int)selected.size());

        // Build direction vectors from p to candidates in this shell
        std::vector<std::pair<Candidate, std::vector<float>>> shell_dirs;
        shell_dirs.reserve(shells[k].size());

        for (auto& cand : shells[k]) {
            const float* c_data = node_data(cand.id);
            std::vector<float> dir(dim_);
            for (int d = 0; d < dim_; ++d)
                dir[d] = c_data[d] - p_data[d];
            shell_dirs.push_back({cand, std::move(dir)});
        }

        // Greedy angular diversity selection within shell
        std::vector<std::vector<float>*> selected_dirs;
        selected_dirs.reserve(budget);

        for (auto& [cand, dir] : shell_dirs) {
            if ((int)selected_dirs.size() >= budget) break;

            // Check angular diversity against already-selected in this shell
            bool diverse = true;
            for (auto* sel_dir : selected_dirs) {
                float cs = cos_sim(dir.data(), sel_dir->data(), dim_);
                if (cs > alpha_) {
                    // Too similar in direction — skip
                    diverse = false;
                    break;
                }
            }

            if (diverse) {
                selected.push_back(cand.id);
                selected_dirs.push_back(&dir);
            }
        }

        // If angular filter was too strict and we got nothing, fall back:
        // just take the nearest candidate in this shell
        if (selected_dirs.empty() && !shells[k].empty()) {
            selected.push_back(shells[k][0].id);
        }
    }

    // Store shell info for the node (using first candidate's r_base)
    // Note: shells_ is updated in build(), not here, to allow const
    return selected;
}

// =============================================================
//  GREEDY SEARCH  (best-first, identical to Vamana beam search)
// =============================================================

std::vector<Candidate> RadialLayerIndex::greedy_search_internal(
    const float* query, int start, int L, int* hops_out) const
{
    std::vector<bool> visited(n_, false);

    // Priority queue: (dist, id) — min-heap by dist
    std::set<Candidate> candidates;  // sorted by dist ascending
    std::set<Candidate> results;

    float d0 = l2_sq(query, node_data(start), dim_);
    Candidate c0{d0, start};
    candidates.insert(c0);
    results.insert(c0);
    visited[start] = true;

    int hops = 0;

    while (!candidates.empty()) {
        // Pop the closest unvisited candidate
        auto it = candidates.begin();
        Candidate cur = *it;
        candidates.erase(it);
        hops++;

        // If cur is worse than our worst result and results is full → stop
        if ((int)results.size() >= L) {
            auto worst = std::prev(results.end());
            if (cur.dist > worst->dist) break;
        }

        // Expand neighbors
        for (int nb : adj_[cur.id]) {
            if (visited[nb]) continue;
            visited[nb] = true;

            float d = l2_sq(query, node_data(nb), dim_);
            Candidate cnb{d, nb};

            // Insert into results if better than worst or results not full
            if ((int)results.size() < L) {
                results.insert(cnb);
                candidates.insert(cnb);
            } else {
                auto worst = std::prev(results.end());
                if (d < worst->dist) {
                    results.erase(worst);
                    results.insert(cnb);
                    candidates.insert(cnb);
                }
            }
        }
    }

    if (hops_out) *hops_out = hops;

    return std::vector<Candidate>(results.begin(), results.end());
}

// =============================================================
//  MEDOID  (start node selection)
// =============================================================

int RadialLayerIndex::compute_medoid(int sample) const {
    int s = std::min(sample, n_);
    // Compute approximate centroid from a sample
    std::vector<float> centroid(dim_, 0.f);
    std::mt19937 rng(42);
    std::vector<int> idx(n_);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(s);

    for (int i : idx) {
        const float* p = node_data(i);
        for (int d = 0; d < dim_; ++d) centroid[d] += p[d];
    }
    for (int d = 0; d < dim_; ++d) centroid[d] /= s;

    // Find the node closest to the centroid
    float best = std::numeric_limits<float>::max();
    int medoid = 0;
    for (int i : idx) {
        float d = l2_sq(centroid.data(), node_data(i), dim_);
        if (d < best) { best = d; medoid = i; }
    }
    return medoid;
}

// =============================================================
//  INTRINSIC DIMENSION ESTIMATE
//  Uses the "two nearest neighbor" method (Levina & Bickel 2004):
//  For each sample point, estimate local ID from ratio of NN distances.
// =============================================================

float RadialLayerIndex::estimate_intrinsic_dim(int sample) const {
    int s = std::min(sample, n_);
    std::mt19937 rng(42);
    std::vector<int> idx(n_);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(s);

    // For each sample, find its 2 nearest neighbors (brute force over sample)
    float sum_log_ratio = 0.f;
    int count = 0;

    for (int i : idx) {
        float d1 = std::numeric_limits<float>::max(); // 1-NN
        float d2 = std::numeric_limits<float>::max(); // 2-NN
        for (int j : idx) {
            if (j == i) continue;
            float d = l2_sq(node_data(i), node_data(j), dim_);
            if (d < d1) { d2 = d1; d1 = d; }
            else if (d < d2) { d2 = d; }
        }
        if (d1 > 1e-12f && d2 > d1) {
            sum_log_ratio += std::log(d2 / d1);
            count++;
        }
    }

    if (count == 0) return (float)dim_;
    // Levina-Bickel: intrinsic_dim ≈ 1 / (mean of log(d2/d1))
    // (Using 2 neighbors; more neighbors gives better estimate)
    float mean_log = sum_log_ratio / count;
    if (mean_log < 1e-6f) return (float)dim_;
    return 1.f / mean_log;
}

// =============================================================
//  BUILD  (one pass)
// =============================================================

void RadialLayerIndex::build_pass(int L_build) {
    // Random insertion order (prevents early-insertion bias)
    std::vector<int> order(n_);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(std::random_device{}());
    std::shuffle(order.begin(), order.end(), rng);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int idx = 0; idx < n_; ++idx) {
        int p = order[idx];
        const float* p_data = node_data(p);

        // 1. Find candidates via greedy search from start node
        auto cands = greedy_search_internal(p_data, start_node_, L_build);

        // Remove self from candidates
        cands.erase(std::remove_if(cands.begin(), cands.end(),
            [p](const Candidate& c){ return c.id == p; }), cands.end());

        // 2. Radial prune to select diverse multi-scale neighbors
        auto neighbors = radial_prune(p, cands);

        // 3. Set forward edges
        {
#ifdef _OPENMP
            std::lock_guard<std::mutex> lock(mtx_[p]);
#endif
            adj_[p] = neighbors;

            // Record shell info for this node
            if (!cands.empty()) {
                shells_[p] = compute_shells(cands[0].dist);
            }
        }

        // 4. Add backward edges (ensure navigability)
        for (int nb : neighbors) {
#ifdef _OPENMP
            std::lock_guard<std::mutex> lock(mtx_[nb]);
#endif
            auto& nb_adj = adj_[nb];
            // Only add if not already present
            if (std::find(nb_adj.begin(), nb_adj.end(), p) == nb_adj.end()) {
                nb_adj.push_back(p);

                // If nb's degree exceeds R, re-prune it
                if ((int)nb_adj.size() > R_) {
                    // Build candidate list from current neighbors
                    std::vector<Candidate> nb_cands;
                    nb_cands.reserve(nb_adj.size());
                    const float* nb_data = node_data(nb);
                    for (int x : nb_adj) {
                        float d = l2_sq(nb_data, node_data(x), dim_);
                        nb_cands.push_back({d, x});
                    }
                    std::sort(nb_cands.begin(), nb_cands.end());
                    nb_adj = radial_prune(nb, nb_cands);
                    if (!nb_cands.empty())
                        shells_[nb] = compute_shells(nb_cands[0].dist);
                }
            }
        }
    }
}

// =============================================================
//  BUILD  (public entry point)
// =============================================================

void RadialLayerIndex::build(
    const std::vector<float>& data, int n, int dim,
    int R, float m, int L_build, float alpha, bool two_pass)
{
    n_     = n;
    dim_   = dim;
    R_     = R;
    m_     = m;
    alpha_ = alpha;

    // Copy data (ensuring alignment)
    data_.resize((size_t)n * dim);
    std::copy(data.begin(), data.end(), data_.begin());

    // Allocate graph
    adj_.assign(n, {});
    shells_.resize(n);
    mtx_ = std::vector<std::mutex>(n);

    // Estimate intrinsic dimension and suggest m if m <= 0
    if (m_ <= 0.f) {
        float id = estimate_intrinsic_dim(1000);
        m_ = std::pow(2.f, 1.f / std::max(1.f, id));
        std::cout << "[RLG] Auto-selected m = " << m_
                  << " (intrinsic dim ≈ " << id << ")\n";
    }

    // Choose medoid as start node
    start_node_ = compute_medoid(2000);
    std::cout << "[RLG] Start node (medoid): " << start_node_ << "\n";

    // Initialize: each node gets a few random neighbors (bootstrap)
    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, n - 1);
        for (int i = 0; i < n; ++i) {
            int nb = dist(rng);
            if (nb != i) adj_[i].push_back(nb);
        }
    }

    std::cout << "[RLG] Build pass 1 ...\n";
    build_pass(L_build);

    if (two_pass) {
        std::cout << "[RLG] Build pass 2 (refinement) ...\n";
        build_pass(L_build);
    }

    std::cout << "[RLG] Build complete. n=" << n << " dim=" << dim
              << " R=" << R << " m=" << m_ << "\n";
}

// =============================================================
//  SEARCH
// =============================================================

std::vector<int> RadialLayerIndex::search(
    const float* query, int K, int L_search) const
{
    int hops = 0;
    auto cands = greedy_search_internal(query, start_node_, L_search, &hops);

    last_avg_candidates = (float)cands.size();
    last_avg_hops       = (float)hops;

    std::vector<int> result;
    result.reserve(K);
    for (int i = 0; i < std::min(K, (int)cands.size()); ++i)
        result.push_back(cands[i].id);
    return result;
}

std::vector<std::vector<int>> RadialLayerIndex::batch_search(
    const std::vector<float>& queries, int nq, int K, int L_search) const
{
    std::vector<std::vector<int>> results(nq);
    float total_cands = 0.f, total_hops = 0.f;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 32) reduction(+:total_cands, total_hops)
#endif
    for (int i = 0; i < nq; ++i) {
        int hops = 0;
        auto cands = greedy_search_internal(
            queries.data() + (size_t)i * dim_, start_node_, L_search, &hops);
        total_cands += cands.size();
        total_hops  += hops;

        results[i].reserve(K);
        for (int j = 0; j < std::min(K, (int)cands.size()); ++j)
            results[i].push_back(cands[j].id);
    }

    last_avg_candidates = total_cands / nq;
    last_avg_hops       = total_hops  / nq;
    return results;
}

// =============================================================
//  DEGREE STATS
// =============================================================

void RadialLayerIndex::degree_stats(
    float& mean, float& stddev, float& gini) const
{
    std::vector<float> degs(n_);
    for (int i = 0; i < n_; ++i) degs[i] = (float)adj_[i].size();

    float s = 0.f;
    for (float d : degs) s += d;
    mean = s / n_;

    float ss = 0.f;
    for (float d : degs) ss += (d - mean) * (d - mean);
    stddev = std::sqrt(ss / n_);

    // Gini coefficient
    std::sort(degs.begin(), degs.end());
    float sum_i = 0.f;
    for (int i = 0; i < n_; ++i) sum_i += (i + 1) * degs[i];
    gini = (2.f * sum_i) / (n_ * s) - (n_ + 1.f) / n_;
}

// =============================================================
//  SAVE / LOAD
// =============================================================

void RadialLayerIndex::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    f.write((char*)&n_,    sizeof(n_));
    f.write((char*)&dim_,  sizeof(dim_));
    f.write((char*)&R_,    sizeof(R_));
    f.write((char*)&m_,    sizeof(m_));
    f.write((char*)&alpha_,sizeof(alpha_));
    f.write((char*)&start_node_, sizeof(start_node_));

    // adjacency
    for (int i = 0; i < n_; ++i) {
        int deg = (int)adj_[i].size();
        f.write((char*)&deg, sizeof(deg));
        f.write((char*)adj_[i].data(), deg * sizeof(int));
    }

    // shell info
    f.write((char*)shells_.data(), n_ * sizeof(ShellInfo));
}

void RadialLayerIndex::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    f.read((char*)&n_,    sizeof(n_));
    f.read((char*)&dim_,  sizeof(dim_));
    f.read((char*)&R_,    sizeof(R_));
    f.read((char*)&m_,    sizeof(m_));
    f.read((char*)&alpha_,sizeof(alpha_));
    f.read((char*)&start_node_, sizeof(start_node_));

    adj_.resize(n_);
    for (int i = 0; i < n_; ++i) {
        int deg;
        f.read((char*)&deg, sizeof(deg));
        adj_[i].resize(deg);
        f.read((char*)adj_[i].data(), deg * sizeof(int));
    }

    shells_.resize(n_);
    f.read((char*)shells_.data(), n_ * sizeof(ShellInfo));
}

} // namespace rlg
