// build_rlg.cpp — Build a Radial Layer Graph index
// Place in graphann/src2/
//
// Usage (mirrors build_index.cpp exactly, adds --m and optional --two_pass):
//   ./build/build_rlg \
//       --data   tmp/sift_base.fbin \
//       --output tmp/sift_rlg.bin   \
//       --R 32 --L 75 --alpha 1.2 --gamma 1.5 --m 2.0
//
//   --alpha  : alpha-RNG diversity (1.0–1.5, SAME meaning as Vamana --alpha)
//   --m      : geometric shell multiplier for Stage 2 (1.5–3.0, default 2.0)
//   --gamma  : degree-overflow multiplier (same as Vamana --gamma)
//   --two_pass: 1 = do two passes (default), 0 = single pass

#include "rlg_index.h"
#include "timer.h"

#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --data <fbin_path> --output <index_path>"
              << " [--R 32] [--L 75] [--alpha 1.2] [--gamma 1.5] [--m 2.0]"
              << " [--two_pass 1]\n";
}

int main(int argc, char** argv) {
    std::string data_path, output_path;
    uint32_t R      = 32;
    uint32_t L      = 75;
    float    alpha  = 1.2f;
    float    gamma  = 1.5f;
    float    m      = 2.0f;
    int      two_pass = 1;   // default: on

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--data"      && i+1 < argc) data_path   = argv[++i];
        else if (arg == "--output"    && i+1 < argc) output_path = argv[++i];
        else if (arg == "--R"         && i+1 < argc) R           = std::atoi(argv[++i]);
        else if (arg == "--L"         && i+1 < argc) L           = std::atoi(argv[++i]);
        else if (arg == "--alpha"     && i+1 < argc) alpha       = std::atof(argv[++i]);
        else if (arg == "--gamma"     && i+1 < argc) gamma       = std::atof(argv[++i]);
        else if (arg == "--m"         && i+1 < argc) m           = std::atof(argv[++i]);
        else if (arg == "--two_pass"  && i+1 < argc) two_pass    = std::atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    }

    if (data_path.empty() || output_path.empty()) {
        print_usage(argv[0]); return 1;
    }

    std::cout << "=== Radial Layer Graph Builder ===\n"
              << "Parameters:\n"
              << "  R         = " << R     << "\n"
              << "  L         = " << L     << "\n"
              << "  alpha     = " << alpha << "  (same as Vamana --alpha)\n"
              << "  gamma     = " << gamma << "\n"
              << "  m         = " << m     << "  (shell multiplier, RLG-specific)\n"
              << "  two_pass  = " << two_pass << "\n"
              << "\n"
              << "Shell budget: " << (int)(R * 0.25f) << " / " << R
              << " slots reserved for multi-scale coverage (Stage 2)\n"
              << "RNG budget:   " << R - (int)(R * 0.25f) << " / " << R
              << " slots for alpha-RNG (Stage 1, same as Vamana)\n\n";

    rlg::RLGIndex index;
    Timer total_timer;
    index.build(data_path, R, L, alpha, m, gamma);
    double build_time = total_timer.elapsed_seconds();

    std::cout << "\nTotal build time: " << build_time << " seconds\n";

    float deg_mean, deg_std, deg_gini;
    index.degree_stats(deg_mean, deg_std, deg_gini);
    std::cout << "Degree stats:\n"
              << "  mean  = " << deg_mean
              << "  (expect ~" << R << " for healthy graph; was 19 with old alpha=0.5)\n"
              << "  stdev = " << deg_std   << "\n"
              << "  gini  = " << deg_gini  << "\n\n";

    index.save(output_path);
    std::cout << "Done.\n";
    return 0;
}