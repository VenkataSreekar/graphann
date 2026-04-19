// build_rlg.cpp
// Drop into graphann/src/
//
// Mirrors build_index.cpp's CLI style exactly.
// Usage (same pattern as build_index):
//   ./build/build_rlg \
//       --data    tmp/sift_base.fbin  \
//       --output  tmp/sift_rlg.bin    \
//       --R 32 --L 75 --m 2.0 --alpha 0.5 --two_pass 1
//
// File format for --data / --output:
//   fbin  : [uint32 npts][uint32 dims][npts*dims float32]   (same as vamana)
//   .bin  : RLGIndex binary (NOT compatible with vamana .bin)

#include "rlg_index.h"
#include "io_utils.h"      // load_bin() already in graphann/include — reuse it
#include "timer.h"         // Timer already in graphann/include — reuse it

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ── tiny arg parser (same pattern used in build_index.cpp) ──────────────────
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

// ── fbin loader (uint32 header, then float data) ────────────────────────────
static std::vector<float> load_fbin(const std::string& path, uint32_t& n, uint32_t& dim) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.read((char*)&n,   sizeof(uint32_t));
    f.read((char*)&dim, sizeof(uint32_t));
    std::vector<float> data((size_t)n * dim);
    f.read((char*)data.data(), data.size() * sizeof(float));
    if (!f) throw std::runtime_error("Truncated file: " + path);
    return data;
}

int main(int argc, char** argv) {
    if (argc < 5 || has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        std::cerr
            << "Usage: build_rlg --data <file.fbin> --output <file.bin>\n"
            << "                 [--R 32] [--L 75] [--m 2.0]\n"
            << "                 [--alpha 0.5] [--two_pass 1]\n\n"
            << "  --R          Max degree per node            (default: 32)\n"
            << "  --L          Build beam width               (default: 75)\n"
            << "  --m          Shell multiplier               (default: 2.0,\n"
            << "                                               use 0 to auto-select)\n"
            << "  --alpha      Angular diversity threshold     (default: 0.5)\n"
            << "  --two_pass   Second refinement pass (0/1)   (default: 1)\n";
        return 1;
    }

    std::string data_path   = get_arg(argc, argv, "--data");
    std::string output_path = get_arg(argc, argv, "--output");
    int   R         = std::stoi(get_arg(argc, argv, "--R",         "32"));
    int   L         = std::stoi(get_arg(argc, argv, "--L",         "75"));
    float m         = std::stof(get_arg(argc, argv, "--m",         "2.0"));
    float alpha     = std::stof(get_arg(argc, argv, "--alpha",     "0.5"));
    bool  two_pass  = std::stoi(get_arg(argc, argv, "--two_pass",  "1")) != 0;

    if (data_path.empty() || output_path.empty()) {
        std::cerr << "Error: --data and --output are required.\n";
        return 1;
    }

    // ── load data ─────────────────────────────────────────────────────────────
    std::cout << "Loading data from: " << data_path << "\n";
    uint32_t n, dim;
    auto data = load_fbin(data_path, n, dim);
    std::cout << "  n=" << n << "  dim=" << dim << "\n\n";

    // ── build ─────────────────────────────────────────────────────────────────
    std::cout << "Building Radial Layer Graph index\n"
              << "  R=" << R << "  L=" << L << "  m=" << m
              << "  alpha=" << alpha << "  two_pass=" << (int)two_pass << "\n";

    rlg::RLGIndex index;

    Timer timer;
    timer.reset();
    index.build(data.data(), (int)n, (int)dim, R, m, L, alpha, two_pass);
    double build_sec = timer.elapsed_se();

    std::cout << "\nBuild time: " << build_sec << " s\n";

    // ── degree stats ──────────────────────────────────────────────────────────
    float deg_mean, deg_std, deg_gini;
    index.degree_stats(deg_mean, deg_std, deg_gini);
    std::cout << "Degree stats:\n"
              << "  mean  = " << deg_mean  << "\n"
              << "  stdev = " << deg_std   << "\n"
              << "  gini  = " << deg_gini  << "\n\n";

    // ── save ──────────────────────────────────────────────────────────────────
    std::cout << "Saving index to: " << output_path << "\n";
    index.save(output_path);
    std::cout << "Done.\n";
    return 0;
}
