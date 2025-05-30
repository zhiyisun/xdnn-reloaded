#include "sgemm_f32f16bf16.h"
#include "debug_print.h"
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <stdexcept>

// Function to perform matrix multiplication C = alpha * A * B + beta * C
// where A is F32, B is FP16, and C is BF16
// Only supports transB=false, M=1, alpha=1, beta=0|1 for the next token calculation
// in transformer attention mechanism
void small_sgemm_f32f16bf16(bool transB, int M, int N, int K, float alpha, const float *A, int lda, const XDNN_FP16 *B, int ldb, float beta, XDNN_BF16 *C, int ldc) {
    DEBUG_PRINT();
    // Check constraints as per header comment
    if (transB) {
        throw std::runtime_error("Not supported for transB=true");
    }
    if (M != 1) {
        throw std::runtime_error("Only supports M=1");
    }
    if (alpha != 1.0f) {
        throw std::runtime_error("alpha must be 1.0");
    }
    if (beta != 0.0f && beta != 1.0f) {
        throw std::runtime_error("beta must be 0 or 1");
    }
    
    // This function is optimized for transB=false, M=1
    // A is 1xK row vector, B is KxN matrix
    // Result C is 1xN row vector

    // Temporary storage for FP32 results
    float* temp_C = new float[N];
    
    // M is always 1 in our case
    const int m = 0;
    
    // If beta is 1.0, we need to load the existing C values
    if (beta == 1.0f) {
        for (int n = 0; n < N; n++) {
            temp_C[n] = static_cast<float>(C[m * ldc + n]);
        }
    } else {
        // If beta is 0, we just zero out the temporary array
        std::memset(temp_C, 0, N * sizeof(float));
    }
    
    // For each column in the output
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        
#ifdef __AVX512F__
        // AVX-512 implementation
        int k = 0;
        
        // Process 16 elements at a time using AVX-512
        for (; k <= K - 16; k += 16) {
            // Load weights from A (already in FP32 format)
            __m512 a_vec = _mm512_loadu_ps(&A[k]);
            
            // Create a vector of FP16 values from B with proper striding
            // We need to manually load each value since they are not contiguous in memory
            XDNN_FP16 b_values[16];
            for (int i = 0; i < 16; i++) {
                b_values[i] = B[(k + i) * ldb + n];
            }
            
            // Convert FP16 to FP32 for computation
            __m256i b_vec_int = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_values));
            __m512 b_vec = _mm512_cvtph_ps(b_vec_int);
            
            // Multiply and accumulate
            __m512 prod = _mm512_mul_ps(a_vec, b_vec);
            sum += _mm512_reduce_add_ps(prod);
        }
        
        // Handle remaining elements
        for (; k < K; k++) {
            sum += A[k] * static_cast<float>(B[k * ldb + n]);
        }
#elif defined(__AVX2__)
        // AVX2 implementation
        int k = 0;
        
        // Process 8 elements at a time using AVX2
        for (; k <= K - 8; k += 8) {
            // Load weights from A
            __m256 a_vec = _mm256_loadu_ps(&A[k]);
            
            // Create a vector of FP16 values from B with proper striding
            // We need to manually load each value since they are not contiguous in memory
            XDNN_FP16 b_values[8];
            for (int i = 0; i < 8; i++) {
                b_values[i] = B[(k + i) * ldb + n];
            }
            
            // Convert FP16 to FP32 for computation
            __m128i b_vec_int = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_values));
            __m256 b_vec = _mm256_cvtph_ps(b_vec_int);
            
            // Multiply and accumulate
            __m256 prod = _mm256_mul_ps(a_vec, b_vec);
            
            // Horizontal sum
            __m128 high = _mm256_extractf128_ps(prod, 1);
            __m128 low = _mm256_castps256_ps128(prod);
            __m128 sum4 = _mm_add_ps(high, low);
            __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
            __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
            sum += _mm_cvtss_f32(sum1);
        }
        
        // Handle remaining elements
        for (; k < K; k++) {
            sum += A[k] * static_cast<float>(B[k * ldb + n]);
        }
#else
        // Scalar implementation
        for (int k = 0; k < K; k++) {
            sum += A[k] * static_cast<float>(B[k * ldb + n]);
        }
#endif
        
        // Store the result
        temp_C[n] = sum;
    }
    
    // Convert FP32 results to BF16 and store in C
    for (int n = 0; n < N; n++) {
        C[m * ldc + n] = static_cast<XDNN_BF16>(temp_C[n]);
    }
    
    // Clean up
    delete[] temp_C;
}