#include "conversion.h"
#include "hgemm.h"
#include "debug_print.h"
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include <cmath>
#include <vector>

// Main hgemm implementation with multi-threading support
void xdnn_hgemm(bool transA, bool transB, int M, int N, int K,
                float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *B, int ldb,
                float beta, XDNN_FP16 *C, int ldc) {
    DEBUG_PRINT();
}

// Pack matrix B for optimized computation
void xdnn_hgemm_packb(bool transB, int N, int K, const XDNN_FP16 *B, int ldb, XDNN_FP16 *packedB) {
    DEBUG_PRINT();
    const int block_size = 64;
    int num_blocks = (N + block_size - 1) / block_size; // Round up division
    
    int packed_idx = 0;
    
    // Process each block of 64 columns (or remaining columns for last block)
    for (int block = 0; block < num_blocks; block++) {
        int block_start = block * block_size;
        int block_end = std::min(block_start + block_size, N);
        int block_width = block_end - block_start;
        
        // Pack this block row by row
        for (int k = 0; k < K; k++) {
            for (int n = block_start; n < block_end; n++) {
                if (!transB) {
                    // B is K×N
                    packedB[packed_idx++] = B[k * ldb + n];
                } else {
                    // B is N×K (transposed)
                    packedB[packed_idx++] = B[n * ldb + k];
                }
            }
        }
    }
}

// Compute hgemm with pre-packed B matrix
void xdnn_hgemm_compute(bool transA, int M, int N, int K,
                        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                        float beta, XDNN_FP16 *C, int ldc) {
    DEBUG_PRINT();
    // DEBUG_PRINT_PARAMS("transA = %d, M = %d, N = %d, K = %d, alpha = %f, lda = %d, beta = %f, ldc = %d\n", transA, M, N, K, alpha, lda, beta, ldc);

    // Check if transA is supported
    if (transA) {
        throw std::runtime_error("transA = true is not currently supported in the xdnn_hgemm_compute implementation");
    }
    
    // Unpack the packed B matrix back to original K×N format
    // This reverses the reference_packb algorithm
    std::vector<XDNN_FP16> B_unpacked(K * N);
    const int block_size = 64;
    int num_blocks = (N + block_size - 1) / block_size; // Round up division
    
    int packed_idx = 0;
    
    // Process each block of 64 columns (or remaining columns for last block)
    for (int block = 0; block < num_blocks; block++) {
        int block_start = block * block_size;
        int block_end = std::min(block_start + block_size, N);
        int block_width = block_end - block_start;
        
        // Unpack this block row by row
        for (int k = 0; k < K; k++) {
            for (int n = block_start; n < block_end; n++) {
                // Unpack to K×N format (original B matrix after transpose)
                B_unpacked[k * N + n] = packedB[packed_idx++];
            }
        }
    }

    // Matrix multiplication with unpacked B (now in K×N format)
    // Note: transA = true is not supported and checked above
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = static_cast<float>(C[m * ldc + n]) * beta;
            for (int k = 0; k < K; k++) {
                // A is M×K format when transA = false (the only supported case)
                float a_val = static_cast<float>(A[m * lda + k]);
                float b_val = static_cast<float>(B_unpacked[k * N + n]);
                sum += alpha * a_val * b_val;
            }

            C[m * ldc + n] = XDNN_FP16(sum);
        }
    }
}

// Compute hgemm with SiLU activation
void xdnn_hgemm_compute_silu(bool transA, int M, int N, int K,
                             float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                             float beta, XDNN_FP16 *C, int ldc) {
    DEBUG_PRINT();
}

// Compute hgemm with GELU activation
void xdnn_hgemm_compute_gelu(bool transA, int M, int N, int K,
                             float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                             float beta, XDNN_FP16 *C, int ldc) {
    DEBUG_PRINT();
}

// Compute hgemm with bias addition
void xdnn_hgemm_compute_biasadd(bool transA, int M, int N, int K,
                               float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                               float beta, XDNN_FP16 *C, int ldc, const float *bias) {
    DEBUG_PRINT();
    // DEBUG_PRINT_PARAMS("transA = %d, M = %d, N = %d, K = %d, alpha = %f, lda = %d, beta = %f, ldc = %d\n", transA, M, N, K, alpha, lda, beta, ldc);
    // Compute regular hgemm
    xdnn_hgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Add bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float v = static_cast<float>(C[i * ldc + j]) + bias[j];
            C[i * ldc + j] = XDNN_FP16(v);
        }
    }
}

// Compute hgemm with bias addition and ReLU activation
void xdnn_hgemm_compute_biasadd_relu(bool transA, int M, int N, int K,
                                     float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                     float beta, XDNN_FP16 *C, int ldc, const float *bias) {
    DEBUG_PRINT();
}

// Compute hgemm with residential connection
void xdnn_hgemm_compute_residential(bool transA, int M, int N, int K,
                                    float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                    float beta, XDNN_FP16 *C, int ldc, const float *bias, const XDNN_FP16 *res, int ldres) {
    DEBUG_PRINT();
}

// Compute hgemm with extended residential connection (assumed to be addition)
void xdnn_hgemm_compute_resext(bool transA, int M, int N, int K,
                                   float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                   float beta, XDNN_FP16 *C, int ldc, const float *bias, float gamma, const XDNN_FP16 *res, int ldres) {
    DEBUG_PRINT();
}

// Compute hgemm with residential multiplication
void xdnn_hgemm_compute_resmul(bool transA, int M, int N, int K,
                                   float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                   float beta, XDNN_FP16 *C, int ldc, const XDNN_FP16 *res, int ldres) {
    DEBUG_PRINT();
}

// ================================================================================
// Below is single thread small hgemm
// ================================================================================
void small_hgemm(int M, int N, int K, const XDNN_FP16 *A, int lda, const XDNN_FP16 *B, int ldb, XDNN_FP16 *C, int ldc) {
    DEBUG_PRINT();
}
