// search_rlg.cpp — Search a Radial Layer Graph index and evaluate recall
// Place in graphann/src2/
//
// Usage (mirrors search_index.cpp exactly):
//   ./build/search_rlg \
//       --index   tmp/sift_rlg.bin        \
//       --data    tmp/sift_base.fbin      \
//       --queries tmp/sift_query.fbin     \
//       --gt      tmp/sift_gt.ibin        \
//       --K 10                            \
//       --L 10,20,30,50,75,100,150,200
//
// Output columns (same as search_index.cpp + QPS column):
//   L | Recall@K | Avg Dist Cmps | Avg Latency (us) | P99 Latency (us) | QPS

#include "rlg_index.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --index <idx> --data <fbin>"
              << " --queries <fbin> --gt <ibin>"
              << " --K <k> --L <l1,l2,...>\n";
}

static std::vector<uint32_t> parse_L_values(const std::string& s) {
    std::vector<uint32_t> vals;
    std::istringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) vals.push_back((uint32_t)std::atoi(tok.c_str()));
    std::sort(vals.begin(), vals.end());
    return vals;
}

// Recall@K: fraction of true top-K found in result
static double compute_recall(const std::vector<uint32_t>& result,
                              const uint32_t* gt, uint32_t K)
{
    uint32_t found = 0;
    for (uint32_t i = 0; i < K && i < result.size(); ++i)
        for (uint32_t j = 0; j < K; ++j)
            if (result[i] == gt[j]) { found++; break; }
    return (double)found / K;
}

int main(int argc, char** argv) {
    std::string index_path, data_path, query_path, gt_path, L_str;
    uint32_t K = 10;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--index"   && i+1 < argc) index_path  = argv[++i];
        else if (arg == "--data"    && i+1 < argc) data_path   = argv[++i];
        else if (arg == "--queries" && i+1 < argc) query_path  = argv[++i];
        else if (arg == "--gt"      && i+1 < argc) gt_path     = argv[++i];
        else if (arg == "--K"       && i+1 < argc) K           = (uint32_t)std::atoi(argv[++i]);
        else if (arg == "--L"       && i+1 < argc) L_str       = argv[++i];
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    }

    if (index_path.empty() || data_path.empty() ||
        query_path.empty() || gt_path.empty() || L_str.empty()) {
        print_usage(argv[0]); return 1;
    }

    auto L_values = parse_L_values(L_str);
    if (L_values.empty()) { std::cerr << "No L values.\n"; return 1; }

    // ── Load index ────────────────────────────────────────────────────────────
    std::cout << "Loading index...\n";
    rlg::RLGIndex index;
    index.load(index_path, data_path);   // same API as VamanaIndex::load

    // ── Load queries ──────────────────────────────────────────────────────────
    std::cout << "Loading queries from " << query_path << "...\n";
    FloatMatrix queries = load_fbin(query_path);
    std::cout << "  Queries: " << queries.npts << " x " << queries.dims << "\n";

    if (queries.dims != index.get_dim()) {
        std::cerr << "Error: query dim (" << queries.dims
                  << ") != index dim (" << index.get_dim() << ")\n";
        return 1;
    }

    // ── Load ground truth ─────────────────────────────────────────────────────
    std::cout << "Loading ground truth from " << gt_path << "...\n";
    IntMatrix gt = load_ibin(gt_path);
    std::cout << "  GT: " << gt.npts << " x " << gt.dims << "\n";

    if (gt.npts != queries.npts) {
        std::cerr << "Error: GT rows != query count\n"; return 1;
    }
    if (gt.dims < K) {
        std::cerr << "Warning: GT has " << gt.dims << " neighbours, K=" << K << "\n";
        K = gt.dims;
    }

    uint32_t nq = queries.npts;

    // ── Header — same columns as search_index.cpp + QPS ──────────────────────
    std::cout << "\n=== Search Results (K=" << K << ") ===\n";
    std::cout << std::setw(8)  << "L"
              << std::setw(14) << "Recall@" + std::to_string(K)
              << std::setw(16) << "Avg Dist Cmps"
              << std::setw(18) << "Avg Latency(us)"
              << std::setw(18) << "P99 Latency(us)"
              << std::setw(14) << "QPS"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    // ── Sweep L values ────────────────────────────────────────────────────────
    for (uint32_t L : L_values) {
        std::vector<double> recalls(nq);
        std::vector<uint32_t> dist_cmps(nq);
        std::vector<double>   latencies(nq);

        #pragma omp parallel for schedule(dynamic, 16)
        for (uint32_t q = 0; q < nq; ++q) {
            rlg::SearchResult res = index.search(queries.row(q), K, L);

            recalls[q]   = compute_recall(res.ids, gt.row(q), K);
            dist_cmps[q] = res.dist_cmps;
            latencies[q] = res.latency_us;
        }

        // Aggregate
        double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / nq;
        double avg_cmps   = (double)std::accumulate(
                                dist_cmps.begin(), dist_cmps.end(), 0ULL) / nq;
        double avg_lat    = std::accumulate(latencies.begin(), latencies.end(), 0.0) / nq;

        // P99
        std::sort(latencies.begin(), latencies.end());
        double p99 = latencies[(size_t)(0.99 * nq)];

        // QPS
        double total_us = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double qps = 1e6 * nq / total_us;

        std::cout << std::setw(8)  << L
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(16) << std::fixed << std::setprecision(1) << avg_cmps
                  << std::setw(18) << std::fixed << std::setprecision(1) << avg_lat
                  << std::setw(18) << std::fixed << std::setprecision(1) << p99
                  << std::setw(14) << std::fixed << std::setprecision(1) << qps
                  << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
