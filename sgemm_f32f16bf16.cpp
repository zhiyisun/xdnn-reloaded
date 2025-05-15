#include "conversion.h"
#include "sgemm_f32f16bf16.h"
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include "intrinsic_ext.h"

// Implementation for small GEMM with FP32 and FP16 inputs and BF16 output
// Designed for attention mechanism: softmax(Q * K^T) * V with mixed precision
void small_sgemm_f32f16bf16(bool transB, int M, int N, int K, float alpha, const float *A, int lda, const XDNN_FP16 *B, int ldb, float beta, XDNN_BF16 *C, int ldc) {
    // This function is optimized for transB=false, M=1
    // Used for attention: A is the attention weights, B is the values matrix
    
    // Special case for M=1
    if (M == 1) {
        if (!transB) {  // B is K x N - this is the expected path
            // Apply beta scaling to C first (convert to FP32 temporarily)
            float temp_C[N];
            if (beta != 0.0f) {
                for (int n = 0; n < N; n++) {
                    temp_C[n] = beta * static_cast<float>(C[n]);
                }
            } else {
                std::memset(temp_C, 0, N * sizeof(float));
            }
            
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
                    
                    // Load FP16 values and convert to FP32
                    __m512 b_vec;
                    
                    // FP16 to FP32 conversion - this would use dedicated instructions in actual implementation
                    float b_fp32[16];
                    for (int i = 0; i < 16; i++) {
                        b_fp32[i] = static_cast<float>(B[(k + i) * ldb + n]);
                    }
                    b_vec = _mm512_loadu_ps(b_fp32);
                    
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
                
                // Apply alpha scaling and add to beta-scaled C
                temp_C[n] += alpha * sum;
            }
            
            // Convert results from FP32 to BF16
            for (int n = 0; n < N; n++) {
                C[n] = temp_C[n];  // Implicit conversion to BF16
            }
        } else {  // B is N x K - not the expected usage
            // Apply beta scaling to C
            float temp_C[N];
            if (beta != 0.0f) {
                for (int n = 0; n < N; n++) {
                    temp_C[n] = beta * static_cast<float>(C[n]);
                }
            } else {
                std::memset(temp_C, 0, N * sizeof(float));
            }
            
            // Compute GEMM
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[k] * static_cast<float>(B[n * ldb + k]);
                }
                temp_C[n] += alpha * sum;
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
                // Apply beta scaling
                float result = (beta != 0.0f) ? beta * static_cast<float>(C[m * ldc + n]) : 0.0f;
                
                // Compute matrix multiplication
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
                
                // Apply alpha and store result
                result += alpha * sum;
                C[m * ldc + n] = result;  // Implicit conversion to BF16
            }
        }
    }
}
