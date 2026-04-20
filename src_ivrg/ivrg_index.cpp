#include "ivrg_index.h"
#include "distance.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ivrg {

// ════════════════════════════════════════════════════════════════════════════
//  K-MEANS ROUTING LAYER
// ════════════════════════════════════════════════════════════════════════════

static std::vector<std::vector<float>>
kmeans_pp_init(const float* data, uint32_t dim,
               const std::vector<uint32_t>& sample_idx,
               uint32_t K, std::mt19937& rng)
{
    uint32_t S = (uint32_t)sample_idx.size();
    std::vector<std::vector<float>> centroids;
    centroids.reserve(K);

    std::uniform_int_distribution<uint32_t> pick(0, S - 1);
    uint32_t first = sample_idx[pick(rng)];
    centroids.push_back({data + (size_t)first * dim,
                         data + (size_t)first * dim + dim});

    std::vector<float> min_d2(S, std::numeric_limits<float>::max());

    for (uint32_t k = 1; k < K; ++k) {
        const std::vector<float>& last = centroids.back();
        for (uint32_t i = 0; i < S; ++i) {
            float d = compute_l2sq(data + (size_t)sample_idx[i] * dim,
                                   last.data(), dim);
            if (d < min_d2[i]) min_d2[i] = d;
        }

        float total = 0.f;
        for (float v : min_d2) total += v;

        std::uniform_real_distribution<float> uni(0.f, total);
        float threshold = uni(rng), cum = 0.f;
        uint32_t chosen = sample_idx[0];
        for (uint32_t i = 0; i < S; ++i) {
            cum += min_d2[i];
            if (cum >= threshold) { chosen = sample_idx[i]; break; }
        }
        centroids.push_back({data + (size_t)chosen * dim,
                             data + (size_t)chosen * dim + dim});
    }
    return centroids;
}

void IVRGIndex::build_routing_layer(uint32_t K,
                                     uint32_t T_iter,
                                     uint32_t kmeans_sample)
{
    K_ = K;
    std::mt19937 rng(42);

    std::vector<uint32_t> sample_idx(npts_);
    std::iota(sample_idx.begin(), sample_idx.end(), 0);
    if (npts_ > kmeans_sample) {
        std::shuffle(sample_idx.begin(), sample_idx.end(), rng);
        sample_idx.resize(kmeans_sample);
    }
    uint32_t S = (uint32_t)sample_idx.size();
    std::cout << "[IVRG] k-means: K=" << K << "  sample=" << S
              << "  iters=" << T_iter << "\n";

    centroids_ = kmeans_pp_init(data_, dim_, sample_idx, K, rng);

    std::vector<uint32_t> assign(S);

    for (uint32_t iter = 0; iter < T_iter; ++iter) {
        #pragma omp parallel for schedule(dynamic, 256)
        for (uint32_t i = 0; i < S; ++i) {
            const float* xi = data_ + (size_t)sample_idx[i] * dim_;
            float best = std::numeric_limits<float>::max();
            uint32_t bk = 0;
            for (uint32_t k = 0; k < K; ++k) {
                float d = compute_l2sq(xi, centroids_[k].data(), dim_);
                if (d < best) { best = d; bk = k; }
            }
            assign[i] = bk;
        }

        std::vector<std::vector<double>> sums(K, std::vector<double>(dim_, 0.0));
        std::vector<uint32_t> counts(K, 0);
        for (uint32_t i = 0; i < S; ++i) {
            uint32_t k = assign[i];
            const float* xi = data_ + (size_t)sample_idx[i] * dim_;
            for (uint32_t d = 0; d < dim_; ++d) sums[k][d] += xi[d];
            counts[k]++;
        }
        for (uint32_t k = 0; k < K; ++k) {
            if (counts[k] == 0) continue;
            for (uint32_t d = 0; d < dim_; ++d)
                centroids_[k][d] = (float)(sums[k][d] / counts[k]);
        }

        if ((iter + 1) % 5 == 0 || iter == T_iter - 1)
            std::cout << "[IVRG] Lloyd iter " << (iter+1) << "/" << T_iter << "\n";
    }

    representatives_.assign(K, sample_idx[0]);
    std::vector<float> best_dist(K, std::numeric_limits<float>::max());

    for (uint32_t i = 0; i < S; ++i) {
        uint32_t k = assign[i];
        const float* xi = data_ + (size_t)sample_idx[i] * dim_;
        float d = compute_l2sq(xi, centroids_[k].data(), dim_);
        if (d < best_dist[k]) {
            best_dist[k] = d;
            representatives_[k] = sample_idx[i];
        }
    }

    std::cout << "[IVRG] Routing layer built ("
              << K << " cells, representatives from " << S << " sampled points).\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  ROUTING
// ════════════════════════════════════════════════════════════════════════════

std::vector<uint32_t> IVRGIndex::route(const float* query, uint32_t np) const {
    struct CD { float d; uint32_t k;
                bool operator<(const CD& o) const { return d < o.d; } };
    std::vector<CD> heap;
    heap.reserve(K_);
    for (uint32_t k = 0; k < K_; ++k)
        heap.push_back({compute_l2sq(query, centroids_[k].data(), dim_), k});

    uint32_t take = std::min(np, K_);
    std::partial_sort(heap.begin(), heap.begin() + take, heap.end());

    std::vector<uint32_t> seeds;
    seeds.reserve(take);
    for (uint32_t i = 0; i < take; ++i)
        seeds.push_back(representatives_[heap[i].k]);
    return seeds;
}

// ════════════════════════════════════════════════════════════════════════════
//  GREEDY BEAM SEARCH (With Timestamp Optimization)
// ════════════════════════════════════════════════════════════════════════════

std::pair<std::vector<IVRGIndex::Candidate>, uint32_t>
IVRGIndex::greedy_search(const float* query, uint32_t L,
                          const std::vector<uint32_t>& seeds) const
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

    // Initialise with all provided seeds (from the routing layer)
    for (uint32_t seed : seeds) {
        if (is_visited(seed)) continue;
        set_visited(seed);
        float d = compute_l2sq(query, get_vec(seed), dim_); 
        dist_cmps++;
        candidate_set.insert({d, seed});
    }

    // Always include medoid as a safety fallback if not already visited
    if (!is_visited(start_node_)) {
        set_visited(start_node_);
        float d = compute_l2sq(query, get_vec(start_node_), dim_); 
        dist_cmps++;
        candidate_set.insert({d, start_node_});
    }

    // Trim to L if seeds overfill the beam width
    while (candidate_set.size() > L)
        candidate_set.erase(std::prev(candidate_set.end()));

    std::set<uint32_t> expanded;

    while (true) {
        uint32_t best = UINT32_MAX;
        for (auto& [d, id] : candidate_set)
            if (!expanded.count(id)) { best = id; break; }
        
        if (best == UINT32_MAX) break;
        expanded.insert(best);

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
//  ROBUST PRUNE
// ════════════════════════════════════════════════════════════════════════════

void IVRGIndex::robust_prune(uint32_t node,
                               std::vector<Candidate>& cands,
                               float alpha, uint32_t R)
{
    cands.erase(std::remove_if(cands.begin(), cands.end(),
                    [node](const Candidate& c){ return c.second == node; }),
                cands.end());
    std::sort(cands.begin(), cands.end());

    std::vector<uint32_t> sel;
    sel.reserve(R);

    for (auto& [d, c] : cands) {
        if (sel.size() >= R) break;
        bool keep = true;
        for (uint32_t s : sel) {
            float dcs = compute_l2sq(get_vec(c), get_vec(s), dim_);
            if (d > alpha * dcs) { keep = false; break; }
        }
        if (keep) sel.push_back(c);
    }

    std::lock_guard<std::mutex> lk(locks_[node]);
    graph_[node] = std::move(sel);
}

// ════════════════════════════════════════════════════════════════════════════
//  MEDOID
// ════════════════════════════════════════════════════════════════════════════

uint32_t IVRGIndex::compute_medoid() const {
    std::vector<double> centroid(dim_, 0.0);
    for (uint32_t i = 0; i < npts_; ++i) {
        const float* v = get_vec(i);
        for (uint32_t d = 0; d < dim_; ++d) centroid[d] += v[d];
    }
    for (double& x : centroid) x /= npts_;

    float best = std::numeric_limits<float>::max(); uint32_t med = 0;
    for (uint32_t i = 0; i < npts_; ++i) {
        float dist = 0.f; const float* v = get_vec(i);
        for (uint32_t d = 0; d < dim_; ++d) {
            float diff = v[d] - (float)centroid[d]; dist += diff * diff;
        }
        if (dist < best) { best = dist; med = i; }
    }
    return med;
}

// ════════════════════════════════════════════════════════════════════════════
//  BUILD PASS
// ════════════════════════════════════════════════════════════════════════════

void IVRGIndex::run_build_pass(const std::vector<uint32_t>& perm,
                                uint32_t R, uint32_t L,
                                float alpha, uint32_t gamma_R)
{
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; ++idx) {
        uint32_t point = perm[idx];

        auto [cands, _dc] = greedy_search(get_vec(point), L, {start_node_});

        {
            std::lock_guard<std::mutex> lk(locks_[point]);
            for (uint32_t nb : graph_[point]) {
                float d = compute_l2sq(get_vec(point), get_vec(nb), dim_);
                cands.push_back({d, nb});
            }
        }

        robust_prune(point, cands, alpha, R);

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
                robust_prune(nb, nb_cands, alpha, R);
            }
        }

        if (idx % 50000 == 0) {
            #pragma omp critical
            std::cout << "\r   Processed " << idx << " / " << npts_
                      << " points" << std::flush;
        }
    }
    std::cout << "\r   Processed " << npts_ << " / " << npts_
              << " points\n";
}

void IVRGIndex::build(const std::string& data_path,
                       uint32_t R, uint32_t L,
                       float alpha, float gamma,
                       uint32_t K_clusters, uint32_t nprobe,
                       uint32_t T_iter, uint32_t kmeans_sample)
{
    nprobe_ = nprobe;

    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts; dim_ = mat.dims;
    data_ = mat.data.release(); owns_data_ = true;
    std::cout << "  Points: " << npts_ << "  Dims: " << dim_ << std::endl;

    if (L < R) L = R;
    graph_.assign(npts_, {});
    locks_ = std::vector<std::mutex>(npts_);

    std::mt19937 rng(42);
    start_node_ = compute_medoid();
    std::cout << "  Start node (medoid): " << start_node_ << std::endl;

    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    uint32_t gamma_R = (uint32_t)(gamma * R);

    std::cout << "\n--- Vamana Pass 1 (alpha=1.0) ---" << std::endl;
    run_build_pass(perm, R, L, 1.0f, gamma_R);
    std::cout << "  Pass 1 complete." << std::endl;

    std::shuffle(perm.begin(), perm.end(), rng);
    std::cout << "\n--- Vamana Pass 2 (alpha=" << alpha << ") ---" << std::endl;
    run_build_pass(perm, R, L, alpha, gamma_R);
    std::cout << "  Pass 2 complete." << std::endl;

    size_t total = 0;
    for (uint32_t i = 0; i < npts_; ++i) total += graph_[i].size();
    std::cout << "  Average out-degree: " << (double)total / npts_ << std::endl;

    build_routing_layer(K_clusters, T_iter, kmeans_sample);
}

SearchResult IVRGIndex::search(const float* query, uint32_t K, uint32_t L) const {
    if (L < K) L = K;
    Timer t;

    auto seeds = route(query, nprobe_);
    auto [cands, dist_cmps] = greedy_search(query, L, seeds);
    double latency = t.elapsed_us();

    SearchResult res;
    res.dist_cmps  = dist_cmps;
    res.latency_us = latency;
    res.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < cands.size(); ++i)
        res.ids.push_back(cands[i].second);
    return res;
}

void IVRGIndex::degree_stats(float& mean, float& stddev, float& gini) const {
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

void IVRGIndex::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    f.write((char*)&npts_,       sizeof(uint32_t));
    f.write((char*)&dim_,        sizeof(uint32_t));
    f.write((char*)&start_node_, sizeof(uint32_t));
    f.write((char*)&K_,          sizeof(uint32_t));
    f.write((char*)&nprobe_,     sizeof(uint32_t));

    for (uint32_t i = 0; i < npts_; ++i) {
        uint32_t deg = (uint32_t)graph_[i].size();
        f.write((char*)&deg, sizeof(uint32_t));
        if (deg > 0)
            f.write((char*)graph_[i].data(), deg * sizeof(uint32_t));
    }

    for (uint32_t k = 0; k < K_; ++k)
        f.write((char*)centroids_[k].data(), dim_ * sizeof(float));
    f.write((char*)representatives_.data(), K_ * sizeof(uint32_t));

    std::cout << "Index saved to " << path << std::endl;
}

void IVRGIndex::load(const std::string& index_path,
                      const std::string& data_path)
{
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts; dim_ = mat.dims;
    data_ = mat.data.release(); owns_data_ = true;

    std::ifstream f(index_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + index_path);

    uint32_t fn, fd;
    f.read((char*)&fn,          sizeof(uint32_t));
    f.read((char*)&fd,          sizeof(uint32_t));
    f.read((char*)&start_node_, sizeof(uint32_t));
    f.read((char*)&K_,          sizeof(uint32_t));
    f.read((char*)&nprobe_,     sizeof(uint32_t));

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

    centroids_.resize(K_, std::vector<float>(dim_));
    for (uint32_t k = 0; k < K_; ++k)
        f.read((char*)centroids_[k].data(), dim_ * sizeof(float));

    representatives_.resize(K_);
    f.read((char*)representatives_.data(), K_ * sizeof(uint32_t));

    std::cout << "Index loaded: " << npts_ << " points, dim=" << dim_
              << ", K=" << K_ << ", nprobe=" << nprobe_
              << ", start=" << start_node_ << std::endl;
}

} // namespace ivrg