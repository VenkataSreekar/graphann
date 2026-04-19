// build_rlg.cpp
// Usage: ./build_rlg <data.fbin> <output.rlg> [R=32] [m=2.0] [L=100] [alpha=0.5] [two_pass=1]
//
// Reads .fbin format: [n:int32][dim:int32][n*dim floats]

#include "radial_layer_index.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static std::vector<float> load_fbin(const std::string& path, int& n, int& dim) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.read((char*)&n,   sizeof(int));
    f.read((char*)&dim, sizeof(int));
    std::vector<float> data((size_t)n * dim);
    f.read((char*)data.data(), data.size() * sizeof(float));
    if (!f) throw std::runtime_error("Truncated file: " + path);
    return data;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: build_rlg <data.fbin> <output.rlg> "
                     "[R=32] [m=2.0] [L=100] [alpha=0.5] [two_pass=1]\n";
        return 1;
    }

    std::string data_path  = argv[1];
    std::string index_path = argv[2];
    int   R        = argc > 3 ? std::stoi(argv[3])   : 32;
    float m        = argc > 4 ? std::stof(argv[4])   : 2.0f;
    int   L        = argc > 5 ? std::stoi(argv[5])   : 100;
    float alpha    = argc > 6 ? std::stof(argv[6])   : 0.5f;
    bool  two_pass = argc > 7 ? (std::stoi(argv[7]) != 0) : true;

    std::cout << "=== Radial Layer Graph Builder ===\n"
              << "  data:      " << data_path  << "\n"
              << "  output:    " << index_path << "\n"
              << "  R="    << R     << "  m=" << m << "\n"
              << "  L="    << L     << "  alpha=" << alpha << "\n"
              << "  two_pass=" << (int)two_pass << "\n\n";

    int n, dim;
    auto data = load_fbin(data_path, n, dim);
    std::cout << "Loaded " << n << " points, dim=" << dim << "\n";

    rlg::RadialLayerIndex idx;

    auto t0 = std::chrono::high_resolution_clock::now();
    idx.build(data, n, dim, R, m, L, alpha, two_pass);
    auto t1 = std::chrono::high_resolution_clock::now();

    double build_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Build time: " << build_sec << " s\n";

    // Degree stats
    float deg_mean, deg_std, deg_gini;
    idx.degree_stats(deg_mean, deg_std, deg_gini);
    std::cout << "Degree stats: mean=" << deg_mean
              << " std="  << deg_std
              << " gini=" << deg_gini << "\n";

    idx.save(index_path);
    std::cout << "Saved index to: " << index_path << "\n";
    return 0;
}
