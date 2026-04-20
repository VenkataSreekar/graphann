// search_ivrg.cpp — Search an IVRG index
// Place in graphann/src_ivrg/
//
// Output format is IDENTICAL to search_index.cpp so results can be
// copy-pasted side-by-side in the report.
//
// Usage:
//   ./build/search_ivrg \
//       --index   tmp/sift_ivrg.bin        \
//       --data    tmp/sift_base.fbin       \
//       --queries tmp/sift_query.fbin      \
//       --gt      tmp/sift_gt.ibin         \
//       --K 10 --L 10,20,30,50,75,100,150,200

#include "ivrg_index.h"
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

static void print_usage(const char* p) {
    std::cerr << "Usage: " << p
              << " --index <bin> --data <fbin> --queries <fbin>"
              << " --gt <ibin> --K <k> --L <l1,l2,...>\n";
}

static std::vector<uint32_t> parse_L(const std::string& s) {
    std::vector<uint32_t> v; std::istringstream ss(s); std::string t;
    while (std::getline(ss, t, ','))
        if (!t.empty()) v.push_back((uint32_t)std::atoi(t.c_str()));
    std::sort(v.begin(), v.end()); return v;
}

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
    std::string idx_path, data_path, query_path, gt_path, L_str;
    uint32_t K = 10;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--index"   && i+1 < argc) idx_path   = argv[++i];
        else if (a == "--data"    && i+1 < argc) data_path  = argv[++i];
        else if (a == "--queries" && i+1 < argc) query_path = argv[++i];
        else if (a == "--gt"      && i+1 < argc) gt_path    = argv[++i];
        else if (a == "--K"       && i+1 < argc) K          = std::atoi(argv[++i]);
        else if (a == "--L"       && i+1 < argc) L_str      = argv[++i];
        else if (a == "--help")                  { print_usage(argv[0]); return 0; }
    }
    if (idx_path.empty() || data_path.empty() ||
        query_path.empty() || gt_path.empty() || L_str.empty()) {
        print_usage(argv[0]); return 1;
    }

    auto Lvals = parse_L(L_str);
    if (Lvals.empty()) return 1;

    // ── Load ─────────────────────────────────────────────────────────────────
    std::cout << "Loading index..." << std::endl;
    ivrg::IVRGIndex index;
    index.load(idx_path, data_path);

    std::cout << "Loading queries from " << query_path << "..." << std::endl;
    FloatMatrix queries = load_fbin(query_path);
    std::cout << "  Queries: " << queries.npts << " x " << queries.dims << std::endl;

    if (queries.dims != index.get_dim()) {
        std::cerr << "Dim mismatch.\n"; return 1;
    }

    std::cout << "Loading ground truth from " << gt_path << "..." << std::endl;
    IntMatrix gt = load_ibin(gt_path);
    std::cout << "  Ground truth: " << gt.npts << " x " << gt.dims << std::endl;

    if (gt.dims < K) K = gt.dims;
    uint32_t nq = queries.npts;

    // ── Header — IDENTICAL to search_index.cpp ────────────────────────────────
    std::cout << "\n=== Search Results (K=" << K << ") ===" << std::endl;
    std::cout << std::setw(8)  << "L"
              << std::setw(14) << "Recall@" + std::to_string(K)
              << std::setw(16) << "Avg Dist Cmps"
              << std::setw(18) << "Avg Latency (us)"
              << std::setw(18) << "P99 Latency (us)"
              << std::endl;
    std::cout << std::string(74, '-') << std::endl;

    for (uint32_t L : Lvals) {
        std::vector<double>   recalls(nq), lats(nq);
        std::vector<uint32_t> cmps(nq);

        #pragma omp parallel for schedule(dynamic, 16)
        for (uint32_t q = 0; q < nq; ++q) {
            ivrg::SearchResult res = index.search(queries.row(q), K, L);
            recalls[q] = compute_recall(res.ids, gt.row(q), K);
            cmps[q]    = res.dist_cmps;
            lats[q]    = res.latency_us;
        }

        double avg_r = std::accumulate(recalls.begin(), recalls.end(), 0.0) / nq;
        double avg_c = (double)std::accumulate(cmps.begin(), cmps.end(), 0ULL) / nq;
        double avg_l = std::accumulate(lats.begin(), lats.end(), 0.0) / nq;
        std::sort(lats.begin(), lats.end());
        double p99 = lats[(size_t)(0.99 * nq)];

        // ── Output row — IDENTICAL column widths to search_index.cpp ────────
        std::cout << std::setw(8)  << L
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_r
                  << std::setw(16) << std::fixed << std::setprecision(1) << avg_c
                  << std::setw(18) << std::fixed << std::setprecision(1) << avg_l
                  << std::setw(18) << std::fixed << std::setprecision(1) << p99
                  << std::endl;
    }

    std::cout << "\nDone." << std::endl;
    return 0;
}