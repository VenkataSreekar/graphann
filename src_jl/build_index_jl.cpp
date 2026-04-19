#include "vamana_index_jl.h"
#include "timer.h"

#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --data <fbin_path>"
              << " --output <index_path>"
              << " [--R <max_degree=32>]"
              << " [--L_build <build_search_list=200>]"
              << " [--alpha <rng_alpha=1.2>]"
              << " [--gamma <degree_multiplier=1.5>]"
              << " [--proj_dim <projected_dim=0>]"
              << std::endl;
    std::cerr << "\n  --proj_dim  Reduce vectors to this many dimensions before"
              << "\n              building (0 = disabled). Recommended: dim/4."
              << "\n              Example: for SIFT1M (128D), use --proj_dim 32."
              << std::endl;
}

int main(int argc, char** argv) {
    std::string data_path, output_path;
    uint32_t R        = 32;
    uint32_t L_build  = 200;
    float    alpha    = 1.2f;
    float    gamma    = 1.5f;
    uint32_t proj_dim = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--data"     && i+1 < argc) data_path   = argv[++i];
        else if (arg == "--output"   && i+1 < argc) output_path = argv[++i];
        else if (arg == "--R"        && i+1 < argc) R           = std::atoi(argv[++i]);
        else if (arg == "--L_build"  && i+1 < argc) L_build     = std::atoi(argv[++i]);
        else if (arg == "--alpha"    && i+1 < argc) alpha       = std::atof(argv[++i]);
        else if (arg == "--gamma"    && i+1 < argc) gamma       = std::atof(argv[++i]);
        else if (arg == "--proj_dim" && i+1 < argc) proj_dim    = std::atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    }

    if (data_path.empty() || output_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== Vamana-JL Index Builder ===" << std::endl;
    std::cout << "  R        = " << R        << std::endl;
    std::cout << "  L_build  = " << L_build  << std::endl;
    std::cout << "  alpha    = " << alpha    << std::endl;
    std::cout << "  gamma    = " << gamma    << std::endl;
    std::cout << "  proj_dim = " << (proj_dim ? std::to_string(proj_dim) : "disabled") << std::endl;

    VamanaIndexJL index;
    Timer total_timer;
    index.build(data_path, R, L_build, alpha, gamma, proj_dim);
    std::cout << "\nTotal build time: " << total_timer.elapsed_seconds()
              << " seconds" << std::endl;
    index.save(output_path);
    std::cout << "Done." << std::endl;
    return 0;
}