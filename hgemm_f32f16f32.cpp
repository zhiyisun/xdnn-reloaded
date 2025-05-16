#include "conversion.h"
#include "hgemm_f32f16f32.h"
#include "intrinsic_ext.h"
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <cmath>

// Constants for block sizes to optimize cache usage
#define HGEMM_MC 64
#define HGEMM_NC 240
#define HGEMM_KC 256

// HGEMM constants for FP16 operations
#define HGEMM_MR 8
#define HGEMM_NR 8

// Helper function to compute SILU activation: x * sigmoid(x)
inline float silu_activate(float x) {
    return x / (1.0f + expf(-x));
}

// Helper function to compute GELU activation
inline float gelu_activate(float x) {
    // Approximation of GELU
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Main HGEMM function that handles different matrix layouts
void xdnn_hgemm_f32f16f32(bool transA, bool transB, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *B, int ldb,
        float beta, float *C, int ldc) {
    
    // Allocate temporary packed B matrix
    size_t packedB_size = (size_t)((N + HGEMM_NR - 1) / HGEMM_NR) * 
                         ((K + HGEMM_KC - 1) / HGEMM_KC) * HGEMM_KC * HGEMM_NR * sizeof(XDNN_FP16);
    XDNN_FP16 *packedB = new XDNN_FP16[packedB_size / sizeof(XDNN_FP16)];
    
    // Pack matrix B for better cache locality
    xdnn_hgemm_f32f16f32_packb(transB, N, K, B, ldb, packedB);
    
    // Compute the matrix multiplication
    xdnn_hgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Free temporary packed matrix
    delete[] packedB;
}

// Pack matrix B for efficient computation
void xdnn_hgemm_f32f16f32_packb(bool transB, int N, int K, const XDNN_FP16 *B, int ldb, XDNN_FP16 *packedB) {
    // Use the block-based packing with default block sizes
    xdnn_hgemm_f32f16f32_packb_block(transB, N, K, B, ldb, packedB, HGEMM_MR, HGEMM_NR);
}

// Block-based packing of matrix B
void xdnn_hgemm_f32f16f32_packb_block(bool transB, int N, int K, const XDNN_FP16 *B, int ldb, 
                                     XDNN_FP16 *packedB, int block_rows, int block_cols) {
    // Pack matrix B into block format for better cache usage during computation
    // The packing order is K-major blocks, then N-major blocks.
    // Within each (K_block, N_block) tile, the data is stored K-minor (row) major.
    int num_k_blocks = (K + block_rows - 1) / block_rows;
    int nb = (N + block_cols - 1) / block_cols; // Number of blocks in N dimension

    // Zero initialize the entire packed buffer
    // The size of packedB should be num_k_blocks * nb * block_rows * block_cols
    memset(packedB, 0, num_k_blocks * nb * block_rows * block_cols * sizeof(XDNN_FP16));

    for (int k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
        int k_start = k_block_idx * block_rows;
        int k_end = std::min(k_start + block_rows, K);

        for (int n_block_idx = 0; n_block_idx < nb; ++n_block_idx) {
            int n_start = n_block_idx * block_cols;
            int n_end = std::min(n_start + block_cols, N);

            // Calculate base offset for the current (K_block, N_block) tile in packedB
            XDNN_FP16* packedB_tile_ptr = packedB + (k_block_idx * nb + n_block_idx) * block_rows * block_cols;

            for (int k_local = 0; k_local < (k_end - k_start); ++k_local) {
                for (int n_local = 0; n_local < (n_end - n_start); ++n_local) {
                    int global_k = k_start + k_local;
                    int global_n = n_start + n_local;
                    XDNN_FP16 val;
                    if (transB) {
                        // Original B is N x K (col-major from C++ perspective if ldb=N, or row-major if ldb=K)
                        // Access B[global_n, global_k]
                        val = B[global_n * ldb + global_k];
                    } else {
                        // Original B is K x N (row-major from C++ perspective if ldb=N)
                        // Access B[global_k, global_n]
                        val = B[global_k * ldb + global_n];
                    }
                    // Store in K-minor (row) major within the tile
                    packedB_tile_ptr[k_local * block_cols + n_local] = val;
                }
            }
        }
    }
}

// Basic HGEMM computation with packed B matrix
void xdnn_hgemm_f32f16f32_compute(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc) {
    
    // Apply beta scaling to C matrix
    if (beta == 0.0f) {
        // If beta is 0, just clear the output matrix
        for (int i = 0; i < M; i++) {
            memset(&C[i * ldc], 0, N * sizeof(float));
        }
    } else if (beta != 1.0f) {
        // If beta is not 1, scale the output matrix
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    // Direct reference implementation for accuracy - corrected to access packedB properly
    const int pack_b_k_block_size = HGEMM_MR; // Block size for K-dimension of B during packing
    const int pack_b_n_block_size = HGEMM_NR; // Block size for N-dimension of B during packing
    const int num_n_blocks_in_packedb = (N + pack_b_n_block_size - 1) / pack_b_n_block_size;

    for (int m = 0; m < M; m++) {
        for (int n_outer = 0; n_outer < N; n_outer++) { // Renamed 'n' to 'n_outer' to avoid conflict
            float sum = 0.0f;
            
            for (int k_outer = 0; k_outer < K; k_outer++) { // Renamed 'k' to 'k_outer'
                float a_val;
                if (transA) {
                    a_val = A[k_outer * lda + m];
                } else {
                    a_val = A[m * lda + k_outer];
                }

                // Calculate indices for accessing packedB
                // B is logically KxN. We need element B[k_outer, n_outer]
                int k_block_major_idx = k_outer / pack_b_k_block_size;
                int k_minor_idx       = k_outer % pack_b_k_block_size;

                int n_block_major_idx = n_outer / pack_b_n_block_size;
                int n_minor_idx       = n_outer % pack_b_n_block_size;
                
                int packed_b_idx = (k_block_major_idx * num_n_blocks_in_packedb + n_block_major_idx) * (pack_b_k_block_size * pack_b_n_block_size) +
                                   k_minor_idx * pack_b_n_block_size + n_minor_idx;

                float b_val = _xdnn_to_float(packedB[packed_b_idx]);
                sum += a_val * b_val;
            }
            
            C[m * ldc + n_outer] += alpha * sum;
        }
    }
}

// HGEMM with SILU activation
void xdnn_hgemm_f32f16f32_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then apply SILU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = silu_activate(C[i * ldc + j]);
        }
    }
}

// HGEMM with GELU activation
void xdnn_hgemm_f32f16f32_compute_gelu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then apply GELU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = gelu_activate(C[i * ldc + j]);
        }
    }
}

// HGEMM with extended residual connection
void xdnn_hgemm_f32f16f32_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc, const float *bias,
        float gamma, const float *res, int ldres) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then add bias and scaled residual
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Add bias if provided
            if (bias) C[i * ldc + j] += bias[j];
            
            // Add scaled residual
            C[i * ldc + j] += gamma * res[i * ldres + j];
        }
    }
}

// HGEMM with residual multiplication (element-wise)
void xdnn_hgemm_f32f16f32_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc, const float *res, int ldres) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then multiply by residual (element-wise)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] *= res[i * ldres + j];
        }
    }
}

// HGEMM with bias addition
void xdnn_hgemm_f32f16f32_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc, const float *bias) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then add bias
    if (bias) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] += bias[j];
            }
        }
    }
}

// HGEMM with bias addition and ReLU activation
void xdnn_hgemm_f32f16f32_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc, const float *bias) {
    
    // First compute HGEMM with bias addition
    xdnn_hgemm_f32f16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
    
    // Then apply ReLU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = std::max(0.0f, C[i * ldc + j]);
        }
    }
}

// HGEMM with bias addition and residual connection
void xdnn_hgemm_f32f16f32_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, float *C, int ldc, const float *bias, const float *res, int ldres) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then add bias and residual
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Add bias if provided
            if (bias) C[i * ldc + j] += bias[j];
            
            // Add residual
            C[i * ldc + j] += res[i * ldres + j];
        }
    }
}

// Small optimized HGEMM implementation for tiny matrices
void small_hgemm_f32f16f32(int M, int N, int K, const float *A, int lda, const XDNN_FP16 *B, int ldb, float *C, int ldc) {
    // First zero initialize output
    for (int i = 0; i < M; i++) {
        memset(&C[i * ldc], 0, N * sizeof(float));
    }
    
    // Basic implementation for small matrices without blocking
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * _xdnn_to_float(B[k * ldb + j]);
            }
            
            C[i * ldc + j] = sum;
        }
    }
}
