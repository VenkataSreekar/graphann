// search_rlg.cpp
// Usage: ./search_rlg <index.rlg> <data.fbin> <queries.fbin> <groundtruth.ibin>
//                     [K=10] [L_search=100]
//
// Outputs: recall@K, QPS, avg_candidates_evaluated, avg_hops
// Also sweeps L_search from 10→500 to produce a full Pareto curve.

#include "radial_layer_index.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <algorithm>

// ---- file loaders ----

static std::vector<float> load_fbin(const std::string& path, int& n, int& dim) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.read((char*)&n,   sizeof(int));
    f.read((char*)&dim, sizeof(int));
    std::vector<float> data((size_t)n * dim);
    f.read((char*)data.data(), data.size() * sizeof(float));
    return data;
}

// groundtruth: [nq:int32][K:int32][nq*K int32s]
static std::vector<std::vector<int>> load_ibin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    int nq, K;
    f.read((char*)&nq, sizeof(int));
    f.read((char*)&K,  sizeof(int));

    std::vector<std::vector<int>> gt(nq, std::vector<int>(K));
    for (int i = 0; i < nq; ++i)
        f.read((char*)gt[i].data(), K * sizeof(int));
    return gt;
}

// ---- recall computation ----
static float compute_recall(
    const std::vector<std::vector<int>>& results,
    const std::vector<std::vector<int>>& groundtruth,
    int K)
{
    int nq = (int)results.size();
    double total = 0.0;
    for (int i = 0; i < nq; ++i) {
        int hits = 0;
        for (int r : results[i]) {
            for (int j = 0; j < K && j < (int)groundtruth[i].size(); ++j) {
                if (r == groundtruth[i][j]) { hits++; break; }
            }
        }
        total += (double)hits / K;
    }
    return (float)(total / nq);
}

// ---- latency percentiles ----
static void percentiles(std::vector<double>& times,
                        double& p50, double& p95, double& p99) {
    std::sort(times.begin(), times.end());
    int n = (int)times.size();
    p50 = times[n * 50 / 100];
    p95 = times[n * 95 / 100];
    p99 = times[n * 99 / 100];
}

// ---- single benchmark run ----
struct RunResult {
    int    L_search;
    float  recall;
    double qps;
    double p50_us, p95_us, p99_us;
    float  avg_cands;
    float  avg_hops;
};

static RunResult benchmark_one(
    const rlg::RadialLayerIndex& idx,
    const std::vector<float>& queries, int nq,
    const std::vector<std::vector<int>>& gt,
    int K, int L_search)
{
    // Per-query latency
    std::vector<double> times(nq);
    std::vector<std::vector<int>> results(nq);

    for (int i = 0; i < nq; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        results[i] = idx.search(queries.data() + (size_t)i * idx.dim(), K, L_search);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    RunResult r;
    r.L_search  = L_search;
    r.recall    = compute_recall(results, gt, K);

    double total_us = std::accumulate(times.begin(), times.end(), 0.0);
    r.qps = 1e6 * nq / total_us;

    percentiles(times, r.p50_us, r.p95_us, r.p99_us);
    r.avg_cands = idx.last_avg_candidates;
    r.avg_hops  = idx.last_avg_hops;

    return r;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: search_rlg <index.rlg> <data.fbin> <queries.fbin> "
                     "<groundtruth.ibin> [K=10] [L=100]\n";
        return 1;
    }

    std::string idx_path   = argv[1];
    std::string data_path  = argv[2];
    std::string query_path = argv[3];
    std::string gt_path    = argv[4];
    int K        = argc > 5 ? std::stoi(argv[5]) : 10;
    int L_single = argc > 6 ? std::stoi(argv[6]) : 100;

    // Load everything
    rlg::RadialLayerIndex idx;
    idx.load(idx_path);

    int n_data, dim_data;
    auto data = load_fbin(data_path, n_data, dim_data);
    // (data stored in index; we just verify dims match)

    int nq, dim_q;
    auto queries = load_fbin(query_path, nq, dim_q);
    auto gt      = load_ibin(gt_path);

    std::cout << "=== Radial Layer Graph Search ===\n"
              << "  n=" << n_data << " dim=" << dim_data
              << " nq=" << nq << " K=" << K << "\n\n";

    // ---- Full Pareto curve sweep ----
    std::vector<int> L_values = {10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200, 300, 500};

    std::cout << std::left
              << std::setw(8)  << "L_search"
              << std::setw(12) << "Recall@K"
              << std::setw(12) << "QPS"
              << std::setw(12) << "p50(us)"
              << std::setw(12) << "p95(us)"
              << std::setw(12) << "p99(us)"
              << std::setw(14) << "AvgCandidates"
              << std::setw(10) << "AvgHops"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (int L : L_values) {
        auto r = benchmark_one(idx, queries, nq, gt, K, L);
        std::cout << std::setw(8)  << r.L_search
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.recall
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.qps
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.p50_us
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.p95_us
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.p99_us
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.avg_cands
                  << std::setw(10) << std::fixed << std::setprecision(1) << r.avg_hops
                  << "\n";

        if (r.recall > 0.9999f) break; // already perfect
    }

    return 0;
}
