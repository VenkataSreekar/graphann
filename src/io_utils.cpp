#include "io_utils.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================================
// Aligned memory allocation
// ============================================================================
// Allocates memory aligned to a 64-byte boundary so that SIMD instructions
// (SSE, AVX, AVX-512) can perform aligned loads without fault or penalty.
// All data matrices returned by load_fbin / load_ibin use this allocator.

static void* aligned_alloc_wrapper(size_t size) {
    size_t aligned_size = (size + 63) & ~(size_t)63;

#ifdef _WIN32
    void* ptr = _aligned_malloc(aligned_size, 64);
#else
    void* ptr = std::aligned_alloc(64, aligned_size);
#endif

    if (!ptr)
        throw std::runtime_error("Failed to allocate " + std::to_string(size) + " bytes");
    return ptr;
}

// ============================================================================
// load_fbin  — load float vectors from a .fbin file
// ============================================================================
// Binary layout:
//   uint32  npts
//   uint32  dims
//   float[npts * dims]   row-major, full precision
//
// The returned buffer is 64-byte aligned for SIMD-friendly access.
// A size sanity check prevents silent truncation if the file is corrupt.

FloatMatrix load_fbin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    uint32_t npts, dims;
    in.read(reinterpret_cast<char*>(&npts), 4);
    in.read(reinterpret_cast<char*>(&dims), 4);
    if (!in.good())
        throw std::runtime_error("Failed to read header from: " + path);

    size_t data_size = (size_t)npts * dims * sizeof(float);

    auto cur = in.tellg();
    in.seekg(0, std::ios::end);
    auto file_size = in.tellg();
    if ((size_t)(file_size - cur) < data_size)
        throw std::runtime_error("File too small: expected " +
            std::to_string(data_size) + " data bytes, file has " +
            std::to_string(file_size - cur) + " after header");
    in.seekg(cur);

    FloatMatrix mat;
    mat.npts = npts;
    mat.dims = dims;
    mat.data = std::unique_ptr<float[], AlignedFree>(
        static_cast<float*>(aligned_alloc_wrapper(data_size)));

    in.read(reinterpret_cast<char*>(mat.data.get()), data_size);
    if (!in.good())
        throw std::runtime_error("Failed to read data from: " + path);

    return mat;
}

// ============================================================================
// load_ibin  — load uint32_t ground-truth neighbor IDs from a .ibin file
// ============================================================================
// Binary layout mirrors .fbin but with uint32_t entries instead of floats.

IntMatrix load_ibin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    uint32_t npts, dims;
    in.read(reinterpret_cast<char*>(&npts), 4);
    in.read(reinterpret_cast<char*>(&dims), 4);
    if (!in.good())
        throw std::runtime_error("Failed to read header from: " + path);

    size_t data_size = (size_t)npts * dims * sizeof(uint32_t);

    auto cur = in.tellg();
    in.seekg(0, std::ios::end);
    auto file_size = in.tellg();
    if ((size_t)(file_size - cur) < data_size)
        throw std::runtime_error("File too small: expected " +
            std::to_string(data_size) + " data bytes, file has " +
            std::to_string(file_size - cur) + " after header");
    in.seekg(cur);

    IntMatrix mat;
    mat.npts = npts;
    mat.dims = dims;
    mat.data = std::unique_ptr<uint32_t[], AlignedFree>(
        static_cast<uint32_t*>(aligned_alloc_wrapper(data_size)));

    in.read(reinterpret_cast<char*>(mat.data.get()), data_size);
    if (!in.good())
        throw std::runtime_error("Failed to read data from: " + path);

    return mat;
}

// ============================================================================
// save_index_sectors  (Section 3.2: DiskANN Index Layout)
// ============================================================================
// Writes the graph index in the DiskANN sector-aligned format so that each
// node's full-precision vector and its neighbor list share a single 4 KB disk
// sector. This enables BeamSearch to retrieve BOTH the coordinates and the
// neighbor list of any node in a SINGLE random disk read (Section 3.5:
// "Implicit Re-Ranking Using Full-Precision Vectors"):
//
//   "When we retrieve the neighborhood of a point during search, we also
//    retrieve the full coordinates of the point without incurring extra disk
//    reads."                                       — DiskANN paper, §3.5
//
// Sector layout for node i (total = SECTOR_LEN bytes):
//   Offset  0                    : float[dims]    — full-precision coordinates
//   Offset  dims*4               : uint32_t       — out-degree
//   Offset  dims*4 + 4           : uint32_t[deg]  — neighbor IDs
//   Offset  dims*4 + 4 + deg*4   : zeros           — padding to SECTOR_LEN
//
// Because every node occupies exactly SECTOR_LEN bytes, the disk offset of
// node i is simply header_size + i * SECTOR_LEN — no in-memory offset table
// is required, saving RAM.
//
// File header (written before the sector array):
//   uint32  npts
//   uint32  dims
//   uint32  start_node
//   uint32  max_degree   (stored so the reader can sanity-check sector size)
//
// Requirement: SECTOR_LEN >= dims*4 + (1 + max_degree)*4
// With SECTOR_LEN=4096 and max_degree=128, dims can be up to ~896.

void save_index_sectors(const std::string& index_path,
                        uint32_t npts, uint32_t dims, uint32_t start_node,
                        uint32_t max_degree,
                        const float* data,
                        const std::vector<std::vector<uint32_t>>& graph) {
    // Validate that the requested layout fits in one sector
    size_t node_bytes = (size_t)dims * sizeof(float)          // coordinates
                      + sizeof(uint32_t)                       // degree field
                      + (size_t)max_degree * sizeof(uint32_t); // neighbor IDs
    if (node_bytes > SECTOR_LEN)
        throw std::runtime_error(
            "SECTOR_LEN (" + std::to_string(SECTOR_LEN) + ") is too small for "
            "dims=" + std::to_string(dims) + ", max_degree=" +
            std::to_string(max_degree) + " (need " +
            std::to_string(node_bytes) + " bytes)");

    std::ofstream out(index_path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open file for writing: " + index_path);

    // Write file header
    out.write(reinterpret_cast<const char*>(&npts),       sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&dims),       sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&start_node), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&max_degree), sizeof(uint32_t));

    // Scratch buffer for one sector (zero-initialised so padding is clean)
    std::vector<char> sector(SECTOR_LEN, 0);

    for (uint32_t i = 0; i < npts; i++) {
        // Zero out the scratch buffer (clears padding from the previous node)
        std::memset(sector.data(), 0, SECTOR_LEN);

        char* ptr = sector.data();

        // Field 1: full-precision coordinate vector
        const float* vec = data + (size_t)i * dims;
        std::memcpy(ptr, vec, dims * sizeof(float));
        ptr += dims * sizeof(float);

        // Field 2: out-degree
        uint32_t deg = static_cast<uint32_t>(graph[i].size());
        std::memcpy(ptr, &deg, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        // Field 3: neighbor IDs
        if (deg > 0)
            std::memcpy(ptr, graph[i].data(), deg * sizeof(uint32_t));

        // Write the padded sector to disk
        out.write(sector.data(), SECTOR_LEN);
    }

    if (!out.good())
        throw std::runtime_error("Failed to write sector-aligned index to: " +
                                  index_path);

    std::cout << "Sector-aligned index saved to " << index_path
              << " (" << npts << " nodes, " << SECTOR_LEN
              << " bytes/sector)" << std::endl;
}

// ============================================================================
// load_node_sector  (used by BeamSearch at query time)
// ============================================================================
// Fetches a single SECTOR_LEN-byte block from the index file for node_id.
// Because each node is exactly one sector, a single fseek + fread retrieves
// both the full-precision coordinates AND the neighbor list.
//
// This is the I/O primitive that allows DiskANN to amortize disk latency:
// BeamSearch calls this for W nodes concurrently in one "round", mapping to W
// parallel async disk reads that arrive in roughly one round-trip latency.
//
// `fp` must be opened with fopen(..., "rb") by the caller and kept open for
// the duration of the search session. Thread safety of concurrent fseek/fread
// calls to the same FILE* is platform-dependent; use one FILE* per thread or
// serialize with a mutex for production use.
//
// Returns the out-degree (number of neighbors stored for this node).

uint32_t load_node_sector(std::FILE* fp,
                          uint32_t node_id,
                          uint32_t dims,
                          float* coord_out,
                          std::vector<uint32_t>& nbrs_out) {
    // Compute the byte offset of node_id's sector.
    // Header is 4 * sizeof(uint32_t) = 16 bytes.
    static constexpr size_t HEADER_BYTES = 4 * sizeof(uint32_t);
    size_t offset = HEADER_BYTES + (size_t)node_id * SECTOR_LEN;

    if (std::fseek(fp, static_cast<long>(offset), SEEK_SET) != 0)
        throw std::runtime_error("fseek failed for node " +
                                  std::to_string(node_id));

    // Read the full sector into a local buffer
    std::vector<char> sector(SECTOR_LEN);
    if (std::fread(sector.data(), 1, SECTOR_LEN, fp) != SECTOR_LEN)
        throw std::runtime_error("fread failed for node " +
                                  std::to_string(node_id));

    const char* ptr = sector.data();

    // Field 1: full-precision coordinates
    std::memcpy(coord_out, ptr, dims * sizeof(float));
    ptr += dims * sizeof(float);

    // Field 2: out-degree
    uint32_t deg = 0;
    std::memcpy(&deg, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    // Field 3: neighbor IDs
    nbrs_out.resize(deg);
    if (deg > 0)
        std::memcpy(nbrs_out.data(), ptr, deg * sizeof(uint32_t));

    return deg;
}