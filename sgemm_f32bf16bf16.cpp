#include "conversion.h"
#include "sgemm_f32bf16bf16.h"
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include "intrinsic_ext.h"

// Implementation for small single-precision GEMM with FP32 and BF16 inputs and BF16 output
// Designed for attention mechanism with softmax(Q * K^T) * V operation
void small_sgemm_f32bf16bf16(bool transB, int M, int N, int K, const float *A, int lda, const XDNN_BF16 *B, int ldb, XDNN_BF16 *C, int ldc) {
    // This function is optimized for transB=false, M=1
    // Processes one row of A (the attention weights) times matrix B (the values)
    
    // Special case for M=1
    if (M == 1) {
        if (!transB) {  // B is K x N - this is the expected path
            // Initialize temporary FP32 results
            float temp_C[N];
            std::memset(temp_C, 0, N * sizeof(float));
            
            // For each column in the output
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                
                // Using AVX-512 for vectorized computation when available
                #ifdef __AVX512F__
                int k = 0;
                // Process 16 elements at a time
                for (; k <= K - 16; k += 16) {
                    // Load attention weights (already FP32)
                    __m512 a_vec = _mm512_loadu_ps(&A[k]);
                    
                    // Load value matrix elements and convert BF16 -> FP32
                    __m512 b_vec = _mm512_loadu_pbh(&B[k * ldb + n]);
                    
                    // Multiply and accumulate
                    __m512 prod = _mm512_mul_ps(a_vec, b_vec);
                    sum += _mm512_reduce_add_ps(prod);
                }
                
                // Process remaining elements
                for (; k < K; k++) {
                    sum += A[k] * static_cast<float>(B[k * ldb + n]);
                }
                #else
                // Scalar implementation
                for (int k = 0; k < K; k++) {
                    sum += A[k] * static_cast<float>(B[k * ldb + n]);
                }
                #endif
                
                temp_C[n] = sum;
            }
            
            // Convert results from FP32 to BF16
            for (int n = 0; n < N; n++) {
                C[n] = temp_C[n];  // Implicit conversion to BF16
            }
        } else {  // B is N x K - not the expected usage
            // Initialize temporary results
            float temp_C[N];
            std::memset(temp_C, 0, N * sizeof(float));
            
            // Compute GEMM
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[k] * static_cast<float>(B[n * ldb + k]);
                }
                temp_C[n] = sum;
            }
            
            // Convert to BF16
            for (int n = 0; n < N; n++) {
                C[n] = temp_C[n];  // Implicit conversion to BF16
            }
        }
    } else {
        // General case implementation for M > 1
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                
                if (!transB) {
                    for (int k = 0; k < K; k++) {
                        sum += A[m * lda + k] * static_cast<float>(B[k * ldb + n]);
                    }
                } else {
                    for (int k = 0; k < K; k++) {
                        sum += A[m * lda + k] * static_cast<float>(B[n * ldb + k]);
                    }
                }
                
                C[m * ldc + n] = sum;  // Implicit conversion to BF16
            }
        }
    }
}

// Implementation for paged attention with block structure
void small_sgemm_f32bf16bf16_b(bool transB, int M, int N, int K, const float *A, int lda, const XDNN_BF16 *B, int ldb, XDNN_BF16 *C, int ldc, int *blockIndices, int blockStride, int blockSize) {
    // This function is designed for M=1
    if (M != 1) return;
    
    // Initialize temporary FP32 results
    float temp_C[N];
    std::memset(temp_C, 0, N * sizeof(float));
    
    // Process each block
    for (int i = 0; i < N; i += blockSize) {
        // Get the block index for this set of columns
        int blockIdx = blockIndices[i / blockSize];
        
        // Compute the base pointer for this block in matrix B
        const XDNN_BF16 *blockB = B + blockIdx * blockStride;
        
        // Process each column in the block
        for (int j = 0; j < blockSize && i + j < N; j++) {
            float sum = 0.0f;
            
            if (!transB) {  // B is K x blockSize within each block - expected path
                // Use SIMD operations when available
                #ifdef __AVX512F__
                int k = 0;
                for (; k <= K - 16; k += 16) {
                    __m512 a_vec = _mm512_loadu_ps(&A[k]);
                    __m512 b_vec = _mm512_loadu_pbh(&blockB[k * ldb + j]);
                    __m512 prod = _mm512_mul_ps(a_vec, b_vec);
                    sum += _mm512_reduce_add_ps(prod);
                }
                
                // Process remaining elements
                for (; k < K; k++) {
                    sum += A[k] * static_cast<float>(blockB[k * ldb + j]);
                }
                #else
                // Scalar fallback
                for (int k = 0; k < K; k++) {
                    sum += A[k] * static_cast<float>(blockB[k * ldb + j]);
                }
                #endif
            } else {  // B is blockSize x K within each block - not expected
                for (int k = 0; k < K; k++) {
                    sum += A[k] * static_cast<float>(blockB[j * ldb + k]);
                }
            }
            
            temp_C[i + j] = sum;
        }
    }
    
    // Convert FP32 to BF16 for the final result
    for (int n = 0; n < N; n++) {
        C[n] = temp_C[n];  // Implicit conversion to BF16
    }
}
