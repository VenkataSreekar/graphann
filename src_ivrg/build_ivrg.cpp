// build_ivrg.cpp — Build an IVRG index
// Place in graphann/src3/
//
// Usage (superset of build_index.cpp flags):
//   ./build/build_ivrg \
//       --data   tmp/sift_base.fbin \
//       --output tmp/sift_ivrg.bin  \
//       --R 32 --L 75 --alpha 1.2 --gamma 1.5 \
//       --K 512 --nprobe 3 --T 15
//
//   --R, --L, --alpha, --gamma : same meaning as build_index.cpp (Vamana)
//   --K        : number of Voronoi clusters  (default: 512)
//   --nprobe   : seeds per query at search time (default: 3)
//   --T        : Lloyd's iterations for k-means (default: 15)

#include "ivrg_index.h"
#include "timer.h"
#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* p) {
    std::cerr << "Usage: " << p
              << " --data <fbin> --output <bin>"
              << " [--R 32] [--L 75] [--alpha 1.2] [--gamma 1.5]"
              << " [--K 512] [--nprobe 3] [--T 15]\n";
}

int main(int argc, char** argv) {
    std::string data_path, output_path;
    uint32_t R      = 32,  L      = 75;
    float    alpha  = 1.2f, gamma = 1.5f;
    uint32_t K      = 512, nprobe = 3, T = 15;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--data"   && i+1 < argc) data_path   = argv[++i];
        else if (a == "--output" && i+1 < argc) output_path = argv[++i];
        else if (a == "--R"      && i+1 < argc) R           = std::atoi(argv[++i]);
        else if (a == "--L"      && i+1 < argc) L           = std::atoi(argv[++i]);
        else if (a == "--alpha"  && i+1 < argc) alpha       = std::atof(argv[++i]);
        else if (a == "--gamma"  && i+1 < argc) gamma       = std::atof(argv[++i]);
        else if (a == "--K"      && i+1 < argc) K           = std::atoi(argv[++i]);
        else if (a == "--nprobe" && i+1 < argc) nprobe      = std::atoi(argv[++i]);
        else if (a == "--T"      && i+1 < argc) T           = std::atoi(argv[++i]);
        else if (a == "--help" || a == "-h")    { print_usage(argv[0]); return 0; }
    }

    if (data_path.empty() || output_path.empty()) {
        print_usage(argv[0]); return 1;
    }

    std::cout << "=== IVRG Index Builder ===\n"
              << "Vamana params: R=" << R << " L=" << L
              << " alpha=" << alpha << " gamma=" << gamma << "\n"
              << "Routing params: K=" << K << " nprobe=" << nprobe
              << " T_iter=" << T << "\n\n";

    ivrg::IVRGIndex index;
    Timer t;
    index.build(data_path, R, L, alpha, gamma, K, nprobe, T, 200000);
    std::cout << "\nTotal build time: " << t.elapsed_seconds() << " s\n";

    float dm, ds, dg;
    index.degree_stats(dm, ds, dg);
    std::cout << "Degree: mean=" << dm << " stdev=" << ds
              << " gini=" << dg << "\n\n";

    index.save(output_path);
    std::cout << "Done.\n";
    return 0;
}
