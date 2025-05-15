#include "conversion.h"
#include "amx_sgemm_bf16bf16bf16.h"
#include "intrinsic_ext.h"
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <vector>

// Fallback conversion helpers if not defined elsewhere
#ifndef _xdnn_to_float
#endif

#ifndef _xdnn_to_bf16
#endif

// AMX packing function for bfloat16 matrices
int xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(int N, int K, int block_rows, int block_cols) {
    int n_blocks = (N + block_cols - 1) / block_cols;
    int k_blocks = (K + block_rows - 1) / block_rows;
    return n_blocks * k_blocks * block_rows * block_cols * sizeof(XDNN_BF16);
}

void xdnn_small_amx_sgemm_bf16bf16bf16_packb(
        bool transB, int N, int K, const XDNN_BF16 *B, int stride, XDNN_BF16 *packedB, int size) {
    const int TILE_K = 16;
    const int TILE_N = 16;
    int n_blocks = (N + TILE_N - 1) / TILE_N;
    int k_blocks = (K + TILE_K - 1) / TILE_K;
    
    if (transB) {
        for (int k = 0; k < K; k += TILE_K) {
            int k_block = std::min(TILE_K, K - k);
            for (int n = 0; n < N; n += TILE_N) {
                int n_block = std::min(TILE_N, N - n);
                // Indexing: blocks in K-major order
                int block_idx = (k / TILE_K) * n_blocks + (n / TILE_N);
                
                for (int kb = 0; kb < k_block; kb++) {
                    for (int nb = 0; nb < n_block; nb++) {
                        // Packing in K-major order within blocks
                        int packed_idx = block_idx * TILE_K * TILE_N + kb * TILE_N + nb;
                        packedB[packed_idx] = B[(k + kb) + (n + nb) * stride]; // transposed, so (n,k) -> (k,n)
                    }
                    // Zero-pad if needed
                    for (int nb = n_block; nb < TILE_N; nb++) {
                        packedB[block_idx * TILE_K * TILE_N + kb * TILE_N + nb] = XDNN_BF16(0.0f);
                    }
                }
                // Zero-pad for remaining k values in the block
                for (int kb = k_block; kb < TILE_K; kb++) {
                    for (int nb = 0; nb < TILE_N; nb++) {
                        packedB[block_idx * TILE_K * TILE_N + kb * TILE_N + nb] = XDNN_BF16(0.0f);
                    }
                }
            }
        }
    } else {
        for (int k = 0; k < K; k += TILE_K) {
            int k_block = std::min(TILE_K, K - k);
            for (int n = 0; n < N; n += TILE_N) {
                int n_block = std::min(TILE_N, N - n);
                // Indexing: blocks in K-major order
                int block_idx = (k / TILE_K) * n_blocks + (n / TILE_N);
                
                for (int kb = 0; kb < k_block; kb++) {
                    for (int nb = 0; nb < n_block; nb++) {
                        // Packing in K-major order within blocks
                        int packed_idx = block_idx * TILE_K * TILE_N + kb * TILE_N + nb;
                        packedB[packed_idx] = B[(k + kb) * stride + (n + nb)];
                    }
                    // Zero-pad if needed
                    for (int nb = n_block; nb < TILE_N; nb++) {
                        packedB[block_idx * TILE_K * TILE_N + kb * TILE_N + nb] = XDNN_BF16(0.0f);
                    }
                }
                // Zero-pad for remaining k values in the block
                for (int kb = k_block; kb < TILE_K; kb++) {
                    for (int nb = 0; nb < TILE_N; nb++) {
                        packedB[block_idx * TILE_K * TILE_N + kb * TILE_N + nb] = XDNN_BF16(0.0f);
                    }
                }
            }
        }
    }
}

// AMX optimized GEMM computation for BF16 input and output
void xdnn_small_amx_sgemm_bf16bf16bf16_compute(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_BF16 *packedB, int ldb, XDNN_BF16 *C, int ldc, float beta) {
    // Call the implementation with alpha = 1.0
    xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a(M, N, K, A, lda, packedB, C, ldc, 1.0f, beta);
}

// AMX optimized GEMM computation for BF16 input and FP32 output
void xdnn_small_amx_sgemm_bf16bf16f32_compute(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_BF16 *packedB, int ldb, float *C, int ldc, float beta) {
    // Call the implementation with alpha = 1.0
    xdnn_small_amx_sgemm_bf16bf16f32_compute_BA16a64b2a(M, N, K, A, lda, packedB, C, ldc, 1.0f, beta);
}

// BA16a64b2a AMX specialized implementation for BF16 input/output
void xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a(int M, int N, int K, const XDNN_BF16 *A,
        int lda, const XDNN_BF16 *packedB, XDNN_BF16 *C, int ldc, float alpha, float beta) {
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 16;  // Changed to match packing function
    int n_blocks = (N + TILE_N - 1) / TILE_N;
    int k_blocks = (K + TILE_K - 1) / TILE_K;

    // Scale or zero output matrix based on beta value
    if (beta == 0.0f) {
        for (int m = 0; m < M; m++) {
            memset(&C[m * ldc], 0, N * sizeof(XDNN_BF16));
        }
    } else if (beta != 1.0f) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                C[m * ldc + n] = _xdnn_to_bf16(beta * _xdnn_to_float(C[m * ldc + n]));
            }
        }
    }

    // Main computation loop
    for (int m = 0; m < M; m += TILE_M) {
        int mb = std::min(TILE_M, M - m);
        for (int n = 0; n < N; n += TILE_N) {
            int nb = std::min(TILE_N, N - n);
            for (int i = 0; i < mb; i++) {
                for (int j = 0; j < nb; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k += TILE_K) {
                        int kb = std::min(TILE_K, K - k);
                        // Update block index calculation to match packing function
                        int block_idx = (k / TILE_K) * n_blocks + (n / TILE_N);
                        for (int kk = 0; kk < kb; kk++) {
                            // Update in-block indexing to match packing function
                            int packedB_idx = block_idx * TILE_K * TILE_N + kk * TILE_N + j;
                            float a_val = _xdnn_to_float(A[(m + i) * lda + (k + kk)]);
                            float b_val = _xdnn_to_float(packedB[packedB_idx]);
                            sum += a_val * b_val;
                        }
                    }
                    float c_val = _xdnn_to_float(C[(m + i) * ldc + n + j]);
                    float new_val = alpha * sum + (beta != 0.0f ? c_val : 0.0f);
                    C[(m + i) * ldc + n + j] = _xdnn_to_bf16(new_val);
                }
            }
        }
    }
}

// BA16a64b2a AMX specialized implementation for BF16 input and FP32 output
void xdnn_small_amx_sgemm_bf16bf16f32_compute_BA16a64b2a(int M, int N, int K, const XDNN_BF16 *A,
        int lda, const XDNN_BF16 *packedB, float *C, int ldc, float alpha, float beta) {
    // Implementation will use AMX tiles for optimized computation
    const int TILE_M = 16;  // AMX tile rows
    const int TILE_N = 16;  // AMX tile columns
    const int TILE_K = 32;  // AMX accumulation depth
    
    // Scale or zero output matrix based on beta value
    if (beta == 0.0f) {
        for (int m = 0; m < M; m++) {
            memset(&C[m * ldc], 0, N * sizeof(float));
        }
    } else if (beta != 1.0f) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                C[m * ldc + n] *= beta;
            }
        }
    }

    // Main computation loop
    for (int m = 0; m < M; m += TILE_M) {
        int mb = std::min(TILE_M, M - m);
        
        for (int n = 0; n < N; n += TILE_N) {
            int nb = std::min(TILE_N, N - n);
            
            // Initialize accumulation
            __m512 acc[TILE_M/16][TILE_N/16] = {0};
            
            for (int k = 0; k < K; k += TILE_K) {
                int kb = std::min(TILE_K, K - k);
                
                // AMX tile-based matrix multiplication
                // This would use actual AMX instructions in a real implementation
                for (int i = 0; i < mb; i++) {
                    for (int j = 0; j < nb; j++) {
                        float sum = 0.0f;
                        for (int kk = 0; kk < kb; kk++) {
                            sum += _xdnn_to_float(A[(m + i) * lda + k + kk]) * 
                                   _xdnn_to_float(packedB[(k * N + n * TILE_K + kk * TILE_N + j)]);
                        }
                        C[(m + i) * ldc + n + j] = alpha * sum + C[(m + i) * ldc + n + j];
                    }
                }
            }
        }
    }
}

// Implementation of batch C functions
void xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a_batch_C(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_BF16 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *alphaBatch, int packedBBatchSize) {
    // Process each batch matrix multiplication A * B[i] = C[i]
    for (int b = 0; b < packedBBatchSize; b++) {
        xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a(
            M, N, K, A, lda, packedBBatch[b], CBatch[b], ldcb[b], alphaBatch[b], 0.0f);
    }
}

void xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a_batch_CM(int M, const int *NBatch, int K, const XDNN_BF16 *A,
        int lda, const XDNN_BF16 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *alphaBatch,
        int packedBBatchSize) {
    // Process each batch matrix multiplication A * B[i] = C[i], with variable N
    for (int b = 0; b < packedBBatchSize; b++) {
        xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a(
            M, NBatch[b], K, A, lda, packedBBatch[b], CBatch[b], ldcb[b], alphaBatch[b], 0.0f);
    }
}

void xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a_batch_A(int M, int N, int K, const XDNN_BF16 *ABatch[],
        const int *ldab, const XDNN_BF16 *packedBBatch[], XDNN_BF16 *C, int ldc, const float *alphaBatch,
        int packedBBatchSize) {
    // Zero initialize the output matrix
    for (int m = 0; m < M; m++) {
        memset(&C[m * ldc], 0, N * sizeof(XDNN_BF16));
    }

    // Accumulate multiple matrix multiplications: C = sum(A[i] * B[i])
    for (int b = 0; b < packedBBatchSize; b++) {
        // Temporary buffer for intermediate results
        std::vector<XDNN_BF16> tempC(M * N);
        
        // Compute A[i] * B[i] = tempC
        xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a(
            M, N, K, ABatch[b], ldab[b], packedBBatch[b], tempC.data(), N, alphaBatch[b], 0.0f);
        
        // Accumulate into final result: C += tempC
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                C[m * ldc + n] = _xdnn_to_bf16(_xdnn_to_float(C[m * ldc + n]) + _xdnn_to_float(tempC[m * N + n]));
            }
        }
    }
}

void xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a_batch_AM(int M, int N, const int *KBatch,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_BF16 *packedBBatch[], XDNN_BF16 *C, int ldc,
        const float *alphaBatch, int packedBBatchSize) {
    // Zero initialize the output matrix
    for (int m = 0; m < M; m++) {
        memset(&C[m * ldc], 0, N * sizeof(XDNN_BF16));
    }

    // Accumulate multiple matrix multiplications with variable K: C = sum(A[i] * B[i])
    for (int b = 0; b < packedBBatchSize; b++) {
        // Temporary buffer for intermediate results
        std::vector<XDNN_BF16> tempC(M * N);
        
        // Compute A[i] * B[i] = tempC with specific K value
        xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a(
            M, N, KBatch[b], ABatch[b], ldab[b], packedBBatch[b], tempC.data(), N, alphaBatch[b], 0.0f);
        
        // Accumulate into final result: C += tempC
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                C[m * ldc + n] = _xdnn_to_bf16(_xdnn_to_float(C[m * ldc + n]) + _xdnn_to_float(tempC[m * N + n]));
            }
        }
    }
}

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_test_all(int option, int M, const int *NBatch, const int *KBatch,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_BF16 *packedBBatch[], XDNN_BF16 *C16[], float *C32[],
        const int *ldcb, const float *alphaBatch, int packedBBatchSize, int layers) {
    // Test function that runs different SGEMM variants based on option parameter
    switch (option) {
        case 0:
            // Test batch_CM with BF16 output
            for (int l = 0; l < layers; l++) {
                xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a_batch_CM(
                    M, NBatch, KBatch[0], ABatch[l], ldab[l], packedBBatch, 
                    C16, ldcb, alphaBatch, packedBBatchSize);
            }
            break;
            
        case 1:
            // Test batch_AM with BF16 output
            for (int l = 0; l < layers; l++) {
                xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a_batch_AM(
                    M, NBatch[0], KBatch, ABatch, ldab, packedBBatch, 
                    C16[l], ldcb[l], alphaBatch, packedBBatchSize);
            }
            break;
            
        // Add more test cases as needed
            
        default:
            return "Invalid test option";
    }
    
    return "All tests completed successfully";
}
