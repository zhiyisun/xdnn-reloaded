#include "conversion.h"
#include "sgemm_bf16bf16f32.h"
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include "intrinsic_ext.h"

// Implementation for small single-precision GEMM with BF16 inputs and FP32 output
void small_sgemm_bf16bf16f32(bool transB, int M, int N, int K, const XDNN_BF16 *A, int lda, const XDNN_BF16 *B, int ldb, float *C, int ldc) {
    // This function is designed for the case where transB=true, M=1
    // Used in attention mechanism for Q * K^T operations
    
    // Specialized implementation for M=1 (single row of A)
    if (M == 1) {
        // Initialize C to zeros
        std::memset(C, 0, N * sizeof(float));
        
        // Convert A from BF16 to FP32 for computation
        float A_fp32[K];
        for (int k = 0; k < K; k++) {
            A_fp32[k] = static_cast<float>(A[k]);
        }
        
        if (transB) {  // B is N x K
            // For each column in C (each row in B)
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                
                // Use AVX-512 when available for better performance
                #ifdef __AVX512F__
                // Process 16 elements at a time using AVX-512
                int k = 0;
                for (; k <= K - 16; k += 16) {
                    __m512 a_vec = _mm512_loadu_ps(&A_fp32[k]);
                    __m512 b_vec = _mm512_loadu_pbh(&B[n * ldb + k]);
                    __m512 prod = _mm512_mul_ps(a_vec, b_vec);
                    sum += _mm512_reduce_add_ps(prod);
                }
                
                // Process remaining elements
                for (; k < K; k++) {
                    sum += A_fp32[k] * static_cast<float>(B[n * ldb + k]);
                }
                #else
                // Scalar implementation
                for (int k = 0; k < K; k++) {
                    sum += A_fp32[k] * static_cast<float>(B[n * ldb + k]);
                }
                #endif
                
                C[n] = sum;
            }
        } else {  // B is K x N
            // For each output element
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                
                // Compute dot product
                for (int k = 0; k < K; k++) {
                    sum += A_fp32[k] * static_cast<float>(B[k * ldb + n]);
                }
                
                C[n] = sum;
            }
        }
    } else {
        // General case implementation (not optimized since function designed for M=1)
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                
                if (transB) {
                    for (int k = 0; k < K; k++) {
                        sum += static_cast<float>(A[m * lda + k]) * static_cast<float>(B[n * ldb + k]);
                    }
                } else {
                    for (int k = 0; k < K; k++) {
                        sum += static_cast<float>(A[m * lda + k]) * static_cast<float>(B[k * ldb + n]);
                    }
                }
                
                C[m * ldc + n] = sum;
            }
        }
    }
}

// Implementation for paged attention
void small_sgemm_bf16bf16f32_b(bool transB, int M, int N, int K, const XDNN_BF16 *A, int lda, const XDNN_BF16 *B, int ldb, float *C, int ldc, int *blockIndices, int blockStride, int blockSize) {
    // Specialized for paged attention with block structure
    // M should be 1 as designed
    if (M != 1) return;
    
    // Convert A to FP32
    float A_fp32[K];
    for (int k = 0; k < K; k++) {
        A_fp32[k] = static_cast<float>(A[k]);
    }
    
    // Initialize output to zero
    std::memset(C, 0, N * sizeof(float));
    
    // Process each block
    for (int i = 0; i < N; i += blockSize) {
        // Get the block index for this set of columns
        int blockIdx = blockIndices[i / blockSize];
        
        // Compute the base pointer for this block in matrix B
        const XDNN_BF16 *blockB = B + blockIdx * blockStride;
        
        // Process each column in the block
        for (int j = 0; j < blockSize && i + j < N; j++) {
            float sum = 0.0f;
            
            if (transB) {
                // Using SIMD for better performance when available
                #ifdef __AVX512F__
                int k = 0;
                for (; k <= K - 16; k += 16) {
                    __m512 a_vec = _mm512_loadu_ps(&A_fp32[k]);
                    __m512 b_vec = _mm512_loadu_pbh(&blockB[j * ldb + k]);
                    __m512 prod = _mm512_mul_ps(a_vec, b_vec);
                    sum += _mm512_reduce_add_ps(prod);
                }
                
                for (; k < K; k++) {
                    sum += A_fp32[k] * static_cast<float>(blockB[j * ldb + k]);
                }
                #else
                // Scalar fallback
                for (int k = 0; k < K; k++) {
                    sum += A_fp32[k] * static_cast<float>(blockB[j * ldb + k]);
                }
                #endif
            } else {
                for (int k = 0; k < K; k++) {
                    sum += A_fp32[k] * static_cast<float>(blockB[k * ldb + j]);
                }
            }
            
            C[i + j] = sum;
        }
    }
}
