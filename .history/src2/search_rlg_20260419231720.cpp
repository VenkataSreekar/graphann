// search_rlg.cpp
// Drop into graphann/src/
//
// Mirrors search_index.cpp's CLI exactly:
//   --index  --data  --queries  --gt  --K  --L (comma-separated)
//
// Usage:
//   ./build/search_rlg \
//       --index   tmp/sift_rlg.bin      \
//       --data    tmp/sift_base.fbin    \
//       --queries tmp/sift_query.fbin   \
//       --gt      tmp/sift_gt.ibin      \
//       --K 10                          \
//       --L 10,20,30,50,75,100,150,200
//
// Output format (tab-separated, one row per L value):
//   L   recall@K   QPS   avg_candidates   p50_us   p95_us   p99_us

#include "rlg_index.h"
#include "timer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ── arg helpers (same as build_rlg / build_index) ───────────────────────────
static std::string get_arg(int argc, char** argv,
                            const std::string& flag,
                            const std::string& def = "") {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == flag) return argv[i + 1];
    return def;
}
static bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

// ── file loaders ─────────────────────────────────────────────────────────────
static std::vector<float> load_fbin(
    const std::string& path, uint32_t& n, uint32_t& dim)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.read((char*)&n,   sizeof(uint32_t));
    f.read((char*)&dim, sizeof(uint32_t));
    std::vector<float> v((size_t)n * dim);
    f.read((char*)v.data(), v.size() * sizeof(float));
    return v;
}

// ibin: [uint32 nq][uint32 K_gt][nq*K_gt uint32]
static std::vector<std::vector<int>> load_ibin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    uint32_t nq, K_gt;
    f.read((char*)&nq,   sizeof(uint32_t));
    f.read((char*)&K_gt, sizeof(uint32_t));
    std::vector<std::vector<int>> gt(nq, std::vector<int>(K_gt));
    for (uint32_t i = 0; i < nq; ++i) {
        std::vector<uint32_t> row(K_gt);
        f.read((char*)row.data(), K_gt * sizeof(uint32_t));
        for (uint32_t j = 0; j < K_gt; ++j) gt[i][j] = (int)row[j];
    }
    return gt;
}

// ── parse comma-separated int list e.g. "10,20,50,100" ──────────────────────
static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(std::stoi(tok));
    return out;
}

// ── recall@K ─────────────────────────────────────────────────────────────────
static float compute_recall(
    const std::vector<std::vector<int>>& results,
    const std::vector<std::vector<int>>& gt, int K)
{
    int nq = (int)results.size();
    double hits = 0.0;
    for (int i = 0; i < nq; ++i) {
        for (int r : results[i]) {
            for (int j = 0; j < K && j < (int)gt[i].size(); ++j) {
                if (r == gt[i][j]) { hits++; break; }
            }
        }
    }
    return (float)(hits / ((double)nq * K));
}

// ── latency percentiles ───────────────────────────────────────────────────────
static void percentiles(std::vector<double>& t,
                         double& p50, double& p95, double& p99) {
    std::sort(t.begin(), t.end());
    int n = (int)t.size();
    p50 = t[n * 50 / 100];
    p95 = t[n * 95 / 100];
    p99 = t[n * 99 / 100];
}

// ═════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    if (argc < 9 || has_flag(argc, argv, "--help")) {
        std::cerr
            << "Usage: search_rlg --index <file.bin> --data <base.fbin>\n"
            << "                  --queries <q.fbin>  --gt <gt.ibin>\n"
            << "                  --K 10  --L 10,20,50,100,200\n";
        return 1;
    }

    std::string index_path  = get_arg(argc, argv, "--index");
    std::string data_path   = get_arg(argc, argv, "--data");
    std::string query_path  = get_arg(argc, argv, "--queries");
    std::string gt_path     = get_arg(argc, argv, "--gt");
    int K                   = std::stoi(get_arg(argc, argv, "--K", "10"));
    std::string L_str       = get_arg(argc, argv, "--L", "10,20,30,50,75,100,150,200");

    auto L_values = parse_int_list(L_str);

    // ── load index ───────────────────────────────────────────────────────────
    std::cout << "Loading RLG index from: " << index_path << "\n";
    rlg::RLGIndex index;
    index.load(index_path);
    std::cout << "  n=" << index.n() << "  dim=" << index.dim() << "\n\n";

    // ── load queries + ground truth ──────────────────────────────────────────
    uint32_t nq, dq, nd, dd;
    auto queries = load_fbin(query_path, nq, dq);
    auto data    = load_fbin(data_path,  nd, dd);  // loaded but not used directly
    auto gt      = load_ibin(gt_path);

    // ── Set data reference in index for search ──────────────────────────────
    index.set_data(data.data());

    std::cout << "Queries: " << nq << "  GT K=" << (gt.empty() ? 0 : (int)gt[0].size()) << "\n\n";

    // ── header ───────────────────────────────────────────────────────────────
    std::cout << std::left
              << std::setw(8)  << "L"
              << std::setw(12) << "Recall@K"
              << std::setw(14) << "QPS"
              << std::setw(16) << "AvgCandidates"
              << std::setw(12) << "p50(us)"
              << std::setw(12) << "p95(us)"
              << std::setw(12) << "p99(us)"
              << "\n";
    std::cout << std::string(86, '-') << "\n";

    // ── sweep L values ───────────────────────────────────────────────────────
    for (int L : L_values) {
        if (L < K) continue;  // skip nonsensical L < K

        std::vector<std::vector<int>> results(nq);
        std::vector<double> times(nq);
        double total_cands = 0.0;

        for (uint32_t i = 0; i < nq; ++i) {
            const float* q = queries.data() + (size_t)i * dq;
            auto t0 = std::chrono::high_resolution_clock::now();
            results[i] = index.search(q, K, L);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            // Note: rlg::RLGIndex::search() uses greedy_search which visits
            // exactly the candidates in the result set — we track set size as proxy.
            total_cands += L;  // conservative proxy; replace with instrumented count
        }

        float  recall = compute_recall(results, gt, K);
        double total_us = 0.0;
        for (double t : times) total_us += t;
        double qps = 1e6 * nq / total_us;
        double p50, p95, p99;
        percentiles(times, p50, p95, p99);

        std::cout << std::left
                  << std::setw(8)  << L
                  << std::setw(12) << std::fixed << std::setprecision(4) << recall
                  << std::setw(14) << std::fixed << std::setprecision(1) << qps
                  << std::setw(16) << std::fixed << std::setprecision(1) << (total_cands / nq)
                  << std::setw(12) << std::fixed << std::setprecision(1) << p50
                  << std::setw(12) << std::fixed << std::setprecision(1) << p95
                  << std::setw(12) << std::fixed << std::setprecision(1) << p99
                  << "\n";

        if (recall >= 0.9999f) {
            std::cout << "  (perfect recall — stopping sweep)\n";
            break;
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
