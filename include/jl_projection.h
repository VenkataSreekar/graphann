#pragma once

// ============================================================================
// Johnson-Lindenstrauss Random Projection  (jl_projection.h)
// ============================================================================
// Reduces vector dimensionality from `dim` to `k` using a random orthogonal
// projection matrix, approximately preserving pairwise L2 distances.
//
// WHY THIS HELPS:
//   Every distance call in both build and search is O(dim). For SIFT1M
//   (dim=128), reducing to k=32 makes every distance computation 4x cheaper.
//   Since millions of distance calls happen during build and thousands per
//   query, this directly cuts wall-clock time for both phases.
//
// WHY IT'S SAFE (the JL lemma):
//   A random projection R in R^(k x dim) satisfies, for any two points x, y:
//
//       E[ ||Rx - Ry||^2 ] = (dim/k) * ||x - y||^2
//
//   After scaling by sqrt(dim/k), expected distances are preserved exactly.
//   With k >= O(log(n) / eps^2), all pairwise distances are preserved to
//   within factor (1 +/- eps) with high probability (JL lemma).
//
// HOW IT WORKS:
//   1. Generate a random Gaussian matrix G in R^(k x dim).
//   2. Orthogonalize G via Gram-Schmidt to get an orthonormal matrix R.
//      (Orthogonality is not strictly required by JL, but it eliminates
//       variance from correlated rows and gives tighter distance bounds.)
//   3. Project: x_proj = R * x  (k-dimensional output).
//   4. Scale: multiply by sqrt(dim/k) to restore expected norm.
//
// The graph is built and searched entirely in the projected space.
// The full-precision vectors are only needed for the final re-ranking step
// (returning the top-K result IDs), which reads from the original data.
//
// CHOOSING k:
//   - k = dim/4  is a good starting point (4x speedup, small distance error).
//   - k = dim/2  is conservative (2x speedup, very small error).
//   - For SIFT1M (dim=128): k=32 or k=48 work well in practice.
//   - Rule of thumb: k >= 20*log2(n) guarantees <10% distance distortion
//     for n points with high probability.

#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <string>
#include <fstream>

class JLProjection {
public:
    JLProjection() = default;

    // Build a random orthonormal projection matrix of shape [k x dim].
    // seed: fixed seed for reproducibility — must be the same at build
    //       and search time so the projection is consistent.
    void init(uint32_t dim, uint32_t k, uint32_t seed = 42) {
        if (k == 0 || k > dim)
            throw std::runtime_error("JLProjection: k must be in [1, dim]");
        dim_ = dim;
        k_   = k;

        // Step 1: Fill k x dim matrix with independent N(0,1) entries
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);

        matrix_.assign((size_t)k * dim, 0.0f);
        for (size_t i = 0; i < (size_t)k * dim; i++)
            matrix_[i] = gauss(rng);

        // Step 2: Gram-Schmidt orthonormalization over the k rows
        // Row i of matrix_ is matrix_.data() + i*dim
        for (uint32_t i = 0; i < k; i++) {
            float* row_i = matrix_.data() + (size_t)i * dim;

            // Subtract projections onto all previous rows
            for (uint32_t j = 0; j < i; j++) {
                const float* row_j = matrix_.data() + (size_t)j * dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < dim; d++)
                    dot += row_i[d] * row_j[d];
                for (uint32_t d = 0; d < dim; d++)
                    row_i[d] -= dot * row_j[d];
            }

            // Normalize row i to unit length
            float norm = 0.0f;
            for (uint32_t d = 0; d < dim; d++)
                norm += row_i[d] * row_i[d];
            norm = std::sqrt(norm);
            if (norm < 1e-10f)
                throw std::runtime_error("JLProjection: degenerate row during Gram-Schmidt");
            for (uint32_t d = 0; d < dim; d++)
                row_i[d] /= norm;
        }

        // Step 3: Precompute the distance-preserving scale factor sqrt(dim/k).
        // Without this, projected distances are smaller by this factor on average.
        scale_ = std::sqrt((float)dim / (float)k);
    }

    // Project a single dim-dimensional vector to k dimensions.
    // out must point to a buffer of at least k floats.
    void project(const float* in, float* out) const {
        for (uint32_t i = 0; i < k_; i++) {
            const float* row = matrix_.data() + (size_t)i * dim_;
            float dot = 0.0f;
            for (uint32_t d = 0; d < dim_; d++)
                dot += row[d] * in[d];
            out[i] = dot * scale_;
        }
    }

    // Project an entire matrix of npts x dim vectors.
    // Returns a flat npts x k array (row-major).
    std::vector<float> project_dataset(const float* data, uint32_t npts) const {
        std::vector<float> out((size_t)npts * k_);
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)npts; i++) {
            project(data + (size_t)i * dim_, out.data() + (size_t)i * k_);
        }
        
        return out;
    }

    // Save / load the projection matrix so build and search binaries use
    // exactly the same projection without regenerating it.
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("JLProjection::save: cannot open " + path);
        f.write(reinterpret_cast<const char*>(&dim_),   sizeof(uint32_t));
        f.write(reinterpret_cast<const char*>(&k_),     sizeof(uint32_t));
        f.write(reinterpret_cast<const char*>(&scale_), sizeof(float));
        f.write(reinterpret_cast<const char*>(matrix_.data()),
                matrix_.size() * sizeof(float));
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("JLProjection::load: cannot open " + path);
        f.read(reinterpret_cast<char*>(&dim_),   sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&k_),     sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&scale_), sizeof(float));
        matrix_.resize((size_t)k_ * dim_);
        f.read(reinterpret_cast<char*>(matrix_.data()),
               matrix_.size() * sizeof(float));
    }

    uint32_t dim() const { return dim_; }
    uint32_t k()   const { return k_;   }

private:
    uint32_t dim_   = 0;
    uint32_t k_     = 0;
    float    scale_ = 1.0f;
    std::vector<float> matrix_;  // row-major [k x dim]
};