#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <cstdlib>

#ifdef _WIN32
#include <malloc.h>  // Required for _aligned_free on Windows
#endif

// ============================================================================
// Cross-platform aligned-memory deleter
// ============================================================================
// Memory returned by aligned_alloc (Linux/Mac) must be freed with std::free.
// Memory from _aligned_malloc (Windows) must be freed with _aligned_free.
// Using this deleter with unique_ptr ensures the correct free path regardless
// of platform, without leaking memory or causing undefined behaviour.

struct AlignedFree {
    void operator()(void* ptr) const {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
};

// ============================================================================
// FloatMatrix — row-major float array with SIMD-friendly alignment
// ============================================================================
// Reads a .fbin file:
//   4 bytes  — npts  (uint32, number of points)
//   4 bytes  — dims  (uint32, number of dimensions per point)
//   npts * dims * 4 bytes — row-major float data
//
// Memory is 64-byte aligned so AVX-512 SIMD loads operate without penalty.
// Ownership is transferred to the caller via unique_ptr + AlignedFree.

struct FloatMatrix {
    std::unique_ptr<float[], AlignedFree> data;
    uint32_t npts;
    uint32_t dims;

    FloatMatrix() : data(nullptr), npts(0), dims(0) {}

    const float* row(uint32_t i) const { return data.get() + (size_t)i * dims; }
    float*       row(uint32_t i)       { return data.get() + (size_t)i * dims; }
};

// ============================================================================
// IntMatrix — row-major uint32_t array (ground-truth neighbor IDs)
// ============================================================================
// Reads a .ibin file: same binary layout as .fbin but with uint32_t entries.
// Used to load ground-truth top-K neighbor IDs for recall evaluation.

struct IntMatrix {
    std::unique_ptr<uint32_t[], AlignedFree> data;
    uint32_t npts;
    uint32_t dims;

    IntMatrix() : data(nullptr), npts(0), dims(0) {}

    const uint32_t* row(uint32_t i) const { return data.get() + (size_t)i * dims; }
    uint32_t*       row(uint32_t i)       { return data.get() + (size_t)i * dims; }
};

// ============================================================================
// DiskANN sector-aligned index layout  (Section 3.2 of the paper)
// ============================================================================
// The DiskANN paper co-locates each node's full-precision vector AND its
// neighbor list in the same 4 KB disk sector. This allows BeamSearch to fetch
// both in a single random read at no extra I/O cost:
//
//   "When we retrieve the neighborhood of a point during search, we also
//    retrieve the full coordinates of the point without incurring extra disk
//    reads. Reading 4KB-aligned disk address into memory is no more expensive
//    than reading 512B."                          — Section 3.5, DiskANN paper
//
// Layout for node i (padded to SECTOR_LEN bytes):
//   [float * dims]        — full-precision coordinate vector
//   [uint32_t]            — out-degree (number of neighbors)
//   [uint32_t * degree]   — neighbor IDs
//   [padding to SECTOR_LEN]
//
// Each node occupies exactly one sector, so the byte offset of node i is
// simply i * SECTOR_LEN — no offset table is needed in memory.
//
// SECTOR_LEN must be at least  dims*4 + (1 + R)*4  bytes.
// 4096 bytes accommodates R=128 in 128 dimensions (512 + 516 = 1028 bytes)
// with room to spare; use a larger value if dims or R exceed this.

static constexpr uint32_t SECTOR_LEN = 4096;

// ============================================================================
// Free functions
// ============================================================================

FloatMatrix load_fbin(const std::string& path);
IntMatrix   load_ibin(const std::string& path);

// Sector-aligned DiskANN index save / load.
// save_index_sectors: writes npts sectors, one per node, to `index_path`.
// load_node_sector:   reads exactly one sector for node `node_id` from disk.
//                     Called at query time by BeamSearch to fetch a node's
//                     coordinates and neighbor list in a single disk read.

void save_index_sectors(const std::string& index_path,
                        uint32_t npts, uint32_t dims, uint32_t start_node,
                        uint32_t max_degree,
                        const float* data,
                        const std::vector<std::vector<uint32_t>>& graph);

// Fills `coord_out` (must be `dims` floats) and `nbrs_out` with the stored
// neighbor IDs. Returns the out-degree of the node.
uint32_t load_node_sector(std::FILE* fp,
                          uint32_t node_id,
                          uint32_t dims,
                          float* coord_out,
                          std::vector<uint32_t>& nbrs_out);