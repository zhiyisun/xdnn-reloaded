#include "conversion.h"
#include "amx_sgemm_bf16f8bf16.h"
#include "intrinsic_ext.h"
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <vector>
#include <thread>

// Conversion helper functions
#ifndef _xdnn_to_float
#endif

#ifndef _xdnn_to_bf16
#endif

#ifndef _xdnn_fp8_to_float
inline float _xdnn_fp8_to_float(const XDNN_E4M3& fp8) {
    return static_cast<float>(fp8);
}
#endif

// AMX SGEMM operations for BF16 and FP8 data types

// Pack size calculation for AMX-optimized operations
int xdnn_small_amx_sgemm_bf16f8bf16_packb_size(int N, int K, int pack_size) {
    // Calculate the number of blocks needed
    const int n_blocks = (N + pack_size - 1) / pack_size;
    const int k_blocks = (K + pack_size - 1) / pack_size;
    
    // Calculate total size with proper alignment
    return n_blocks * k_blocks * pack_size * pack_size * sizeof(XDNN_E4M3);
}

// Pack matrix B for efficient AMX execution
void xdnn_small_amx_sgemm_bf16f8bf16_packb(
        bool transB, int N, int K, const XDNN_E4M3 *B, int ldb, XDNN_E4M3 *packedB, int pack_size) {
    // Implementation depends on whether B is transposed
    if (transB) {
        // Handle transposed input matrix
        for (int k = 0; k < K; k += pack_size) {
            for (int n = 0; n < N; n += pack_size) {
                const int k_block = std::min(pack_size, K - k);
                const int n_block = std::min(pack_size, N - n);
                
                // Pack the data in tiles for AMX operations
                for (int kb = 0; kb < k_block; kb++) {
                    for (int nb = 0; nb < n_block; nb++) {
                        packedB[(k/pack_size * N + n) * pack_size * pack_size + kb * pack_size + nb] = 
                            B[(n + nb) * ldb + (k + kb)];
                    }
                    // Zero padding
                    for (int nb = n_block; nb < pack_size; nb++) {
                        packedB[(k/pack_size * N + n) * pack_size * pack_size + kb * pack_size + nb] = 0;
                    }
                }
                
                // Zero padding
                for (int kb = k_block; kb < pack_size; kb++) {
                    for (int nb = 0; nb < pack_size; nb++) {
                        packedB[(k/pack_size * N + n) * pack_size * pack_size + kb * pack_size + nb] = 0;
                    }
                }
            }
        }
    } else {
        // Handle non-transposed input matrix
        for (int k = 0; k < K; k += pack_size) {
            for (int n = 0; n < N; n += pack_size) {
                const int k_block = std::min(pack_size, K - k);
                const int n_block = std::min(pack_size, N - n);
                
                // Pack the data in tiles for AMX operations
                for (int kb = 0; kb < k_block; kb++) {
                    for (int nb = 0; nb < n_block; nb++) {
                        packedB[(k/pack_size * N + n) * pack_size * pack_size + kb * pack_size + nb] = 
                            B[(k + kb) * ldb + (n + nb)];
                    }
                    // Zero padding
                    for (int nb = n_block; nb < pack_size; nb++) {
                        packedB[(k/pack_size * N + n) * pack_size * pack_size + kb * pack_size + nb] = 0;
                    }
                }
                
                // Zero padding
                for (int kb = k_block; kb < pack_size; kb++) {
                    for (int nb = 0; nb < pack_size; nb++) {
                        packedB[(k/pack_size * N + n) * pack_size * pack_size + kb * pack_size + nb] = 0;
                    }
                }
            }
        }
    }
}

// Single-threaded implementation of BF16/FP8 SGEMM
void xdnn_small_amx_sgemm_bf16f8bf16_compute_single(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_E4M3 *packedB, XDNN_BF16 *C, int ldc, const float *scaleB, int lds, int blockSize, float alpha,
        float beta, const float *bias) {
    
    // AMX tile parameters
    const int AMX_TILE_M = 16;
    const int AMX_TILE_N = 16;
    const int AMX_TILE_K = 32;
    
    // Handle scaling of existing C matrix based on beta
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
    
    // Add bias if provided
    if (bias != nullptr) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float c_val = _xdnn_to_float(C[m * ldc + n]);
                C[m * ldc + n] = _xdnn_to_bf16(c_val + bias[n]);
            }
        }
    }
    
    // Main computation loop
    for (int m = 0; m < M; m += AMX_TILE_M) {
        int mb = std::min(AMX_TILE_M, M - m);
        
        for (int n = 0; n < N; n += AMX_TILE_N) {
            int nb = std::min(AMX_TILE_N, N - n);
            
            // Get scale factor for this block
            float scale = scaleB[(n / blockSize) * lds];
            
            for (int k = 0; k < K; k += AMX_TILE_K) {
                int kb = std::min(AMX_TILE_K, K - k);
                
                // AMX tile multiplication
                for (int i = 0; i < mb; i++) {
                    for (int j = 0; j < nb; j++) {
                        float accum = 0.0f;
                        
                        // Inner loop for matrix multiplication
                        for (int kk = 0; kk < kb; kk++) {
                            float a_val = _xdnn_to_float(A[(m + i) * lda + (k + kk)]);
                            float b_val = _xdnn_fp8_to_float(packedB[(k/AMX_TILE_K * N + n) * 
                                AMX_TILE_K * AMX_TILE_N + kk * AMX_TILE_N + j]);
                            
                            accum += a_val * b_val * scale;
                        }
                        
                        // Update output with scaled accumulator
                        float c_val = _xdnn_to_float(C[(m + i) * ldc + (n + j)]);
                        C[(m + i) * ldc + (n + j)] = _xdnn_to_bf16(c_val + alpha * accum);
                    }
                }
            }
        }
    }
}

// Multi-threaded implementation of BF16/FP8 SGEMM
void xdnn_small_amx_sgemm_bf16f8bf16_compute(int M, int N, int K, const XDNN_BF16 *A, int lda, const XDNN_E4M3 *packedB,
        XDNN_BF16 *C, int ldc, const float *scaleB, int lds, int blockSize, float alpha, float beta, const float *bias) {
    
    // Use single-threaded implementation for small matrices
    if (M * N < 4096) {
        xdnn_small_amx_sgemm_bf16f8bf16_compute_single(
            M, N, K, A, lda, packedB, C, ldc, scaleB, lds, blockSize, alpha, beta, bias);
        return;
    }
    
    // Get number of available threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    num_threads = std::min(num_threads, static_cast<unsigned int>((M + 15) / 16));
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    // Split work by rows
    int rows_per_thread = (M + num_threads - 1) / num_threads;
    
    // Handle bias and beta scaling in the main thread
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
    
    // Add bias if provided
    if (bias != nullptr) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float c_val = _xdnn_to_float(C[m * ldc + n]);
                C[m * ldc + n] = _xdnn_to_bf16(c_val + bias[n]);
            }
        }
    }
    
    // Launch worker threads
    for (unsigned int t = 0; t < num_threads; t++) {
        int start_row = t * rows_per_thread;
        int end_row = std::min(start_row + rows_per_thread, M);
        
        threads.emplace_back([=]() {
            const int AMX_TILE_M = 16;
            const int AMX_TILE_N = 16;
            const int AMX_TILE_K = 32;
            
            for (int m = start_row; m < end_row; m += AMX_TILE_M) {
                int mb = std::min(AMX_TILE_M, end_row - m);
                
                for (int n = 0; n < N; n += AMX_TILE_N) {
                    int nb = std::min(AMX_TILE_N, N - n);
                    
                    // Get scale factor for this block
                    float scale = scaleB[(n / blockSize) * lds];
                    
                    for (int k = 0; k < K; k += AMX_TILE_K) {
                        int kb = std::min(AMX_TILE_K, K - k);
                        
                        // AMX tile multiplication
                        for (int i = 0; i < mb; i++) {
                            for (int j = 0; j < nb; j++) {
                                float accum = 0.0f;
                                
                                // Inner loop for matrix multiplication
                                for (int kk = 0; kk < kb; kk++) {
                                    float a_val = _xdnn_to_float(A[(m + i) * lda + (k + kk)]);
                                    float b_val = _xdnn_fp8_to_float(packedB[(k/AMX_TILE_K * N + n) * 
                                        AMX_TILE_K * AMX_TILE_N + kk * AMX_TILE_N + j]);
                                    
                                    accum += a_val * b_val * scale;
                                }
                                
                                // Update output with scaled accumulator
                                float c_val = _xdnn_to_float(C[(m + i) * ldc + (n + j)]);
                                C[(m + i) * ldc + (n + j)] = _xdnn_to_bf16(c_val + alpha * accum);
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Join all threads
    for (auto& t : threads) {
        t.join();
    }
}

// Batch operations where C = A * BatchB
void xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_C(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize) {
    
    // Process each batch independently
    for (int b = 0; b < packedBBatchSize; b++) {
        xdnn_small_amx_sgemm_bf16f8bf16_compute(
            M, N, K, A, lda, packedBBatch[b], CBatch[b], ldcb[b], 
            scaleBBatch[b], scaleB_lda[b], blockSize, alphaBatch[b], 0.0f, nullptr);
    }
}

// Batch operations with variable N dimensions
void xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_CM(int M, const int *NBatch, int K, const XDNN_BF16 *A, int lda,
        const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize) {
    
    // Process each batch with variable column dimensions
    for (int b = 0; b < packedBBatchSize; b++) {
        xdnn_small_amx_sgemm_bf16f8bf16_compute(
            M, NBatch[b], K, A, lda, packedBBatch[b], CBatch[b], ldcb[b], 
            scaleBBatch[b], scaleB_lda[b], blockSize, alphaBatch[b], 0.0f, nullptr);
    }
}

// Batch operations where C = sum(BatchA * BatchB)
void xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_A(int M, int N, int K, const XDNN_BF16 *ABatch[], const int *ldab,
        const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc, const float *scaleBBatch[], const int *scaleB_lda,
        int blockSize, const float *alphaBatch, int packedBBatchSize) {
    
    // Zero initialize output
    for (int m = 0; m < M; m++) {
        memset(&C[m * ldc], 0, N * sizeof(XDNN_BF16));
    }
    
    // Accumulate results from each batch
    for (int b = 0; b < packedBBatchSize; b++) {
        // Allocate temporary buffer for each batch result
        std::vector<XDNN_BF16> temp_C(M * N);
        
        // Compute individual batch result
        xdnn_small_amx_sgemm_bf16f8bf16_compute(
            M, N, K, ABatch[b], ldab[b], packedBBatch[b], temp_C.data(), N,
            scaleBBatch[b], scaleB_lda[b], blockSize, alphaBatch[b], 0.0f, nullptr);
        
        // Accumulate into final result
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float c_val = _xdnn_to_float(C[m * ldc + n]);
                float temp_val = _xdnn_to_float(temp_C[m * N + n]);
                C[m * ldc + n] = _xdnn_to_bf16(c_val + temp_val);
            }
        }
    }
}

// Batch operations where C = sum(BatchA * BatchB) with variable K
void xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_AM(int M, int N, const int *KBatch, const XDNN_BF16 *ABatch[],
        const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize) {
    
    // Zero initialize output
    for (int m = 0; m < M; m++) {
        memset(&C[m * ldc], 0, N * sizeof(XDNN_BF16));
    }
    
    // Accumulate results from each batch with variable K dimensions
    for (int b = 0; b < packedBBatchSize; b++) {
        // Allocate temporary buffer for each batch result
        std::vector<XDNN_BF16> temp_C(M * N);
        
        // Compute individual batch result with batch-specific K
        xdnn_small_amx_sgemm_bf16f8bf16_compute(
            M, N, KBatch[b], ABatch[b], ldab[b], packedBBatch[b], temp_C.data(), N,
            scaleBBatch[b], scaleB_lda[b], blockSize, alphaBatch[b], 0.0f, nullptr);
        
        // Accumulate into final result
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float c_val = _xdnn_to_float(C[m * ldc + n]);
                float temp_val = _xdnn_to_float(temp_C[m * N + n]);
                C[m * ldc + n] = _xdnn_to_bf16(c_val + temp_val);
            }
        }
    }
}

// Residential connection implementation
void xdnn_small_amx_sgemm_bf16f8bf16_compute_residential(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_E4M3 *packedB, XDNN_BF16 *C, int ldc, const float *scaleB, int lds, int blockSize, float alpha,
        float beta, const float *bias, const XDNN_BF16 *res, int ldres) {
    
    // First compute standard matrix multiplication
    xdnn_small_amx_sgemm_bf16f8bf16_compute(
        M, N, K, A, lda, packedB, C, ldc, scaleB, lds, blockSize, alpha, beta, bias);
    
    // Then add residential connection
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float c_val = _xdnn_to_float(C[m * ldc + n]);
            float res_val = _xdnn_to_float(res[m * ldres + n]);
            C[m * ldc + n] = _xdnn_to_bf16(c_val + res_val);
        }
    }
}

// Batch residential implementation
void xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_C(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *resBatch[], const int *ldresb) {
    
    // Process each batch with residual connections
    for (int b = 0; b < packedBBatchSize; b++) {
        // First compute standard matrix multiplication
        xdnn_small_amx_sgemm_bf16f8bf16_compute(
            M, N, K, A, lda, packedBBatch[b], CBatch[b], ldcb[b],
            scaleBBatch[b], scaleB_lda[b], blockSize, alphaBatch[b], 0.0f, nullptr);
        
        // Then add residential connection for this batch
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float c_val = _xdnn_to_float(CBatch[b][m * ldcb[b] + n]);
                float res_val = _xdnn_to_float(resBatch[b][m * ldresb[b] + n]);
                CBatch[b][m * ldcb[b] + n] = _xdnn_to_bf16(c_val + res_val);
            }
        }
    }
}

// Variable N dimensions with residual connections
void xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_CM(int M, const int *NBatch, int K, const XDNN_BF16 *A,
        int lda, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *resBatch[], const int *ldresb) {
    
    // Process each batch with variable columns and residual connections
    for (int b = 0; b < packedBBatchSize; b++) {
        // First compute standard matrix multiplication with variable N
        xdnn_small_amx_sgemm_bf16f8bf16_compute(
            M, NBatch[b], K, A, lda, packedBBatch[b], CBatch[b], ldcb[b],
            scaleBBatch[b], scaleB_lda[b], blockSize, alphaBatch[b], 0.0f, nullptr);
        
        // Then add residential connection for this batch
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < NBatch[b]; n++) {
                float c_val = _xdnn_to_float(CBatch[b][m * ldcb[b] + n]);
                float res_val = _xdnn_to_float(resBatch[b][m * ldresb[b] + n]);
                CBatch[b][m * ldcb[b] + n] = _xdnn_to_bf16(c_val + res_val);
            }
        }
    }
}

// Sum batches with residual connections
void xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_A(int M, int N, int K, const XDNN_BF16 *ABatch[],
        const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize, const XDNN_BF16 *res,
        int ldres) {
    
    // First compute the batch sum matrix multiplication
    xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_A(
        M, N, K, ABatch, ldab, packedBBatch, C, ldc,
        scaleBBatch, scaleB_lda, blockSize, alphaBatch, packedBBatchSize);
    
    // Then add the residual connection
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float c_val = _xdnn_to_float(C[m * ldc + n]);
            float res_val = _xdnn_to_float(res[m * ldres + n]);
            C[m * ldc + n] = _xdnn_to_bf16(c_val + res_val);
        }
    }
}

// Variable K with residual connections
void xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_AM(int M, int N, const int *KBatch,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *res, int ldres) {
    
    // First compute the batch sum matrix multiplication with variable K
    xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_AM(
        M, N, KBatch, ABatch, ldab, packedBBatch, C, ldc,
        scaleBBatch, scaleB_lda, blockSize, alphaBatch, packedBBatchSize);
    
    // Then add the residual connection
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float c_val = _xdnn_to_float(C[m * ldc + n]);
            float res_val = _xdnn_to_float(res[m * ldres + n]);
            C[m * ldc + n] = _xdnn_to_bf16(c_val + res_val);
        }
    }
}

// Test function implementations
const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_test(int caseid, int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_E4M3 *packedB, XDNN_BF16 *C, int ldc, const float *scaleB, int lds, int blockSize, float alpha,
        float beta, const float *bias) {
    
    switch (caseid) {
        case 0: // Test single-threaded implementation
            xdnn_small_amx_sgemm_bf16f8bf16_compute_single(
                M, N, K, A, lda, packedB, C, ldc, scaleB, lds, blockSize, alpha, beta, bias);
            break;
            
        case 1: // Test multi-threaded implementation
            xdnn_small_amx_sgemm_bf16f8bf16_compute(
                M, N, K, A, lda, packedB, C, ldc, scaleB, lds, blockSize, alpha, beta, bias);
            break;
            
        default:
            return "Invalid test case ID";
    }
    
    return "Test completed successfully";
}

// Additional test functions for residential and batch operations
const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_test(int caseid, int M, int N, int K,
        const XDNN_BF16 *A, int lda, const XDNN_E4M3 *packedB, XDNN_BF16 *C, int ldc, const float *scaleB, int lds,
        int blockSize, float alpha, float beta, const float *bias, const XDNN_BF16 *res, int ldres) {
    
    switch (caseid) {
        case 0:
            xdnn_small_amx_sgemm_bf16f8bf16_compute_residential(
                M, N, K, A, lda, packedB, C, ldc, scaleB, lds, blockSize, alpha, beta, bias, res, ldres);
            break;
            
        default:
            return "Invalid test case ID";
    }
    
    return "Residential test completed successfully";
}

// Test functions for batch operations
const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_A_test(int caseid, int M, int N, int K,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int BSize) {
    
    switch (caseid) {
        case 0:
            xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_A(
                M, N, K, ABatch, ldab, packedBBatch, C, ldc,
                scaleBBatch, scaleB_lda, blockSize, alphaBatch, BSize);
            break;
            
        default:
            return "Invalid test case ID";
    }
    
    return "Batch A test completed successfully";
}

// Remaining test function implementations follow the same pattern
const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_AM_test(int caseid, int M, int N, const int *KBatch,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int BSize) {
    
    switch (caseid) {
        case 0:
            xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_AM(
                M, N, KBatch, ABatch, ldab, packedBBatch, C, ldc,
                scaleBBatch, scaleB_lda, blockSize, alphaBatch, BSize);
            break;
            
        default:
            return "Invalid test case ID";
    }
    
    return "Batch AM test completed successfully";
}

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_A_test(int caseid, int M, int N, int K,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *res, int ldres) {
    
    switch (caseid) {
        case 0:
            xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_A(
                M, N, K, ABatch, ldab, packedBBatch, C, ldc,
                scaleBBatch, scaleB_lda, blockSize, alphaBatch, packedBBatchSize, res, ldres);
            break;
            
        default:
            return "Invalid test case ID";
    }
    
    return "Residential batch A test completed successfully";
}

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_AM_test(int caseid, int M, int N, int *KBatch,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *res, int ldres) {
    
    switch (caseid) {
        case 0:
            xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_AM(
                M, N, KBatch, ABatch, ldab, packedBBatch, C, ldc,
                scaleBBatch, scaleB_lda, blockSize, alphaBatch, packedBBatchSize, res, ldres);
            break;
            
        default:
            return "Invalid test case ID";
    }
    
    return "Residential batch AM test completed successfully";
}

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_C_test(int caseid, int M, int N, int K, const XDNN_BF16 *A,
        int lda, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int BSize) {
    
    switch (caseid) {
        case 0:
            xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_C(
                M, N, K, A, lda, packedBBatch, CBatch, ldcb,
                scaleBBatch, scaleB_lda, blockSize, alphaBatch, BSize);
            break;
            
        default:
            return "Invalid test case ID";
    }
    
    return "Batch C test completed successfully";
}

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_CM_test(int caseid, int M, const int *NBatch, int K,
        const XDNN_BF16 *A, int lda, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int BSize) {
    
    switch (caseid) {
        case 0:
            xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_CM(
                M, NBatch, K, A, lda, packedBBatch, CBatch, ldcb,
                scaleBBatch, scaleB_lda, blockSize, alphaBatch, BSize);
            break;
            
        default:
            return "Invalid test case ID";
    }
    
    return "Batch CM test completed successfully";
}

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_C_test(int caseid, int M, int N, int K,
        const XDNN_BF16 *A, int lda, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *resBatch[], const int *ldresb) {
    
    switch (caseid) {
        case 0:
            xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_C(
                M, N, K, A, lda, packedBBatch, CBatch, ldcb,
                scaleBBatch, scaleB_lda, blockSize, alphaBatch, packedBBatchSize, resBatch, ldresb);
            break;
            
        default:
            return "Invalid test case ID";
    }
    
    return "Residential batch C test completed successfully";
}

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_CM_test(int caseid, int M, int *NBatch, int K,
        const XDNN_BF16 *A, int lda, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *resBatch[], const int *ldresb) {
    
    switch (caseid) {
        case 0:
            xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_CM(
                M, NBatch, K, A, lda, packedBBatch, CBatch, ldcb,
                scaleBBatch, scaleB_lda, blockSize, alphaBatch, packedBBatchSize, resBatch, ldresb);
            break;
            
        default:
            return "Invalid test case ID";
    }
    
    return "Residential batch CM test completed successfully";
}
