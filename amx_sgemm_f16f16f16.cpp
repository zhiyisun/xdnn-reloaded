#include "conversion.h"
#include "amx_sgemm_f16f16f16.h"
#include "intrinsic_ext.h"
#include <cstring>
#include <immintrin.h>
#include <algorithm>

// AMX SGEMM operations for FP16 data types

int xdnn_small_amx_sgemm_f16f16f16_packb_size(int N, int K, int block_rows, int block_cols) {
    // Calculate required size for packing including alignment padding
    // Each block is stored contiguously for better memory access patterns
    int num_blocks = (N + block_cols - 1) / block_cols * (K + block_rows - 1) / block_rows;
    return num_blocks * block_rows * block_cols * sizeof(XDNN_FP16);
}

void xdnn_small_amx_sgemm_f16f16f16_packb(bool transB, int N, int K, const XDNN_FP16 *B, int stride, XDNN_FP16 *packedB,
                                          int size) {
    // Define tile size for AMX operations
    const int amx_tile_rows = 16;
    const int amx_tile_cols = 16;
    
    if (transB) {
        // Handle transposed input - extract in column-major format and store in blocks
        for (int k = 0; k < K; k += amx_tile_rows) {
            for (int n = 0; n < N; n += amx_tile_cols) {
                const int k_block = std::min(amx_tile_rows, K - k);
                const int n_block = std::min(amx_tile_cols, N - n);
                
                for (int kb = 0; kb < k_block; kb++) {
                    for (int nb = 0; nb < n_block; nb++) {
                        packedB[(k * N + n * amx_tile_rows + kb * amx_tile_cols + nb)] = 
                            B[(n + nb) * stride + (k + kb)];
                    }
                    // Pad to full tile width if needed
                    for (int nb = n_block; nb < amx_tile_cols; nb++) {
                        packedB[(k * N + n * amx_tile_rows + kb * amx_tile_cols + nb)] = 0;
                    }
                }
                
                // Pad to full tile height if needed
                for (int kb = k_block; kb < amx_tile_rows; kb++) {
                    for (int nb = 0; nb < amx_tile_cols; nb++) {
                        packedB[(k * N + n * amx_tile_rows + kb * amx_tile_cols + nb)] = 0;
                    }
                }
            }
        }
    } else {
        // Handle non-transposed input - extract in row-major format and store in blocks
        for (int k = 0; k < K; k += amx_tile_rows) {
            for (int n = 0; n < N; n += amx_tile_cols) {
                const int k_block = std::min(amx_tile_rows, K - k);
                const int n_block = std::min(amx_tile_cols, N - n);
                
                for (int kb = 0; kb < k_block; kb++) {
                    for (int nb = 0; nb < n_block; nb++) {
                        packedB[(k * N + n * amx_tile_rows + kb * amx_tile_cols + nb)] = 
                            B[(k + kb) * stride + (n + nb)];
                    }
                    // Pad to full tile width if needed
                    for (int nb = n_block; nb < amx_tile_cols; nb++) {
                        packedB[(k * N + n * amx_tile_rows + kb * amx_tile_cols + nb)] = 0;
                    }
                }
                
                // Pad to full tile height if needed
                for (int kb = k_block; kb < amx_tile_rows; kb++) {
                    for (int nb = 0; nb < amx_tile_cols; nb++) {
                        packedB[(k * N + n * amx_tile_rows + kb * amx_tile_cols + nb)] = 0;
                    }
                }
            }
        }
    }
}

// AMX optimized GEMM computation for FP16 input and output
void xdnn_small_amx_sgemm_f16f16f16_compute(int M, int N, int K, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                            int ldb, XDNN_FP16 *C, int ldc, float beta) {
    // AMX tile parameters for optimized computation
    const int TILE_M = 16;  // AMX tile rows
    const int TILE_N = 16;  // AMX tile columns
    const int TILE_K = 32;  // AMX accumulation depth
    
    // Scale or zero output matrix based on beta value
    if (beta == 0.0f) {
        for (int m = 0; m < M; m++) {
            memset(&C[m * ldc], 0, N * sizeof(XDNN_FP16));
        }
    } else if (beta != 1.0f) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                C[m * ldc + n] = _xdnn_to_fp16(beta * _xdnn_to_float(C[m * ldc + n]));
            }
        }
    }

    // Main computation loop - block-based for efficient cache utilization
    for (int m = 0; m < M; m += TILE_M) {
        int mb = std::min(TILE_M, M - m);
        
        for (int n = 0; n < N; n += TILE_N) {
            int nb = std::min(TILE_N, N - n);
            
            // Initialize accumulation registers for this tile
            __m512 acc[TILE_M/16][TILE_N/16] = {0};
            
            for (int k = 0; k < K; k += TILE_K) {
                int kb = std::min(TILE_K, K - k);
                
                // AMX tile-based matrix multiplication
                // This would use actual AMX instructions in a real implementation
                // Here we simulate the computation using scalar operations
                for (int i = 0; i < mb; i++) {
                    for (int j = 0; j < nb; j++) {
                        float sum = 0.0f;
                        for (int kk = 0; kk < kb; kk++) {
                            sum += _xdnn_to_float(A[(m + i) * lda + k + kk]) * 
                                   _xdnn_to_float(packedB[(k * N + n * TILE_K + kk * TILE_N + j)]);
                        }
                        C[(m + i) * ldc + n + j] = _xdnn_to_fp16(sum + 
                            (beta != 0.0f ? _xdnn_to_float(C[(m + i) * ldc + n + j]) : 0.0f));
                    }
                }
            }
        }
    }
}
