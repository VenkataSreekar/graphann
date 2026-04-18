#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <cstdlib>

#ifdef _WIN32
#include <malloc.h> // Required for _aligned_free on Windows
#endif

// Cross-platform custom deleter for aligned memory
struct AlignedFree {
    void operator()(void* ptr) const {
#ifdef _WIN32
        _aligned_free(ptr); // Windows/MinGW requires its own free
#else
        std::free(ptr);     // Linux/Mac uses standard free
#endif
    }
};

// Reads a .fbin file: 4 bytes npts (uint32), 4 bytes dims (uint32),
// then npts * dims floats in row-major order.
// Returns aligned memory for SIMD-friendly access.
// Caller receives ownership via unique_ptr with custom deleter.
struct FloatMatrix {
    std::unique_ptr<float[], AlignedFree> data;
    uint32_t npts;
    uint32_t dims;

    FloatMatrix() : data(nullptr), npts(0), dims(0) {}

    const float* row(uint32_t i) const { return data.get() + (size_t)i * dims; }
    float* row(uint32_t i)       { return data.get() + (size_t)i * dims; }
};

// Reads a .ibin file: same layout but with uint32_t entries.
// Used for ground truth (top-K neighbor IDs per query).
struct IntMatrix {
    std::unique_ptr<uint32_t[], AlignedFree> data;
    uint32_t npts;
    uint32_t dims;

    IntMatrix() : data(nullptr), npts(0), dims(0) {}

    const uint32_t* row(uint32_t i) const { return data.get() + (size_t)i * dims; }
    uint32_t* row(uint32_t i)       { return data.get() + (size_t)i * dims; }
};

FloatMatrix load_fbin(const std::string& path);
IntMatrix   load_ibin(const std::string& path);