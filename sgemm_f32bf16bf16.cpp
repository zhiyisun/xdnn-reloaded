#include "sgemm_f32bf16bf16.h"
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <stdexcept>

// Function to perform matrix multiplication C = A * B
// where A is F32, B is BF16, and C is BF16
// Only supports transB=false, M=1 for the next token calculation
// in transformer attention mechanism
void small_sgemm_f32bf16bf16(bool transB, int M, int N, int K, const float *A, int lda, const XDNN_BF16 *B, int ldb, XDNN_BF16 *C, int ldc) {
    // Check constraints as per header comment
    if (transB) {
        throw std::runtime_error("Not supported for transB=true");
    }
    if (M != 1) {
        throw std::runtime_error("Only supports M=1");
    }
    
    // This function is optimized for transB=false, M=1
    // A is 1xK row vector, B is KxN matrix
    // Result C is 1xN row vector

    // Temporary storage for FP32 results
    float* temp_C = new float[N];
    std::memset(temp_C, 0, N * sizeof(float));
    
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
            
            // Create a vector of BF16 values from B with proper striding
            // We need to manually load each value since they are not contiguous in memory
            XDNN_BF16 b_values[16];
            for (int i = 0; i < 16; i++) {
                b_values[i] = B[(k + i) * ldb + n];
            }
            
            // Convert BF16 to FP32 for computation
            float b_float[16];
            for (int i = 0; i < 16; i++) {
                b_float[i] = static_cast<float>(b_values[i]);
            }
            
            // Load the converted values
            __m512 b_vec = _mm512_loadu_ps(b_float);
            
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
            
            // Create a vector of BF16 values from B with proper striding
            // We need to manually load each value since they are not contiguous in memory
            XDNN_BF16 b_values[8];
            for (int i = 0; i < 8; i++) {
                b_values[i] = B[(k + i) * ldb + n];
            }
            
            // Convert BF16 to FP32 for computation
            float b_float[8];
            for (int i = 0; i < 8; i++) {
                b_float[i] = static_cast<float>(b_values[i]);
            }
            
            // Load the converted values
            __m256 b_vec = _mm256_loadu_ps(b_float);
            
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
        C[n] = static_cast<XDNN_BF16>(temp_C[n]);
    }
    
    // Clean up
    delete[] temp_C;
}

/**
 * This function is specially designed for paged attention
 * Matrix B is like (blockSize=4):
 *                                   |<---- ldb ----->|
 *  ________________ ________________|_h0_|___________|_h0_|___________
 * | #head*headSize | #head*headSize |    |           |                | block0
 * |________________|________________|____|___________|________________|
 * |                |                |                |                | block1
 * |________________|________________|________________|________________|
 * |                |                |                |                | block2
 * |________________|________________|________________|________________|
 * |<--------------------------- blockStride ------------------------->|
 */
void small_sgemm_f32bf16bf16_b(bool transB, int M, int N, int K, const float *A, int lda, 
                               const XDNN_BF16 *B, int ldb, XDNN_BF16 *C, int ldc, 
                               int *blockIndices, int blockStride, int blockSize) {
    // Check constraints
    if (transB) {
        throw std::runtime_error("Not supported for transB=true");
    }
    if (M != 1) {
        throw std::runtime_error("Only supports M=1");
    }
    
    // Zero out the output array
    for (int n = 0; n < N; n++) {
        C[n] = static_cast<XDNN_BF16>(0.0f);
    }
    
    // Calculate number of blocks
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // For each block
    for (int b = 0; b < numBlocks; b++) {
        // Get the block index from indices array
        int blockIdx = blockIndices[b];
        
        // Determine start position in the output
        int outputOffset = b * blockSize;
        
        // Determine start position in B based on block index
        const XDNN_BF16* blockB = B + blockIdx * blockStride;
        
        // Process each column in this block (but not exceeding N)
        int columnsInThisBlock = std::min(blockSize, N - outputOffset);
        
        for (int j = 0; j < columnsInThisBlock; j++) {
            float sum = 0.0f;
            
#ifdef __AVX512F__
            // AVX-512 implementation
            int k = 0;
            
            // Process 16 elements at a time
            for (; k <= K - 16; k += 16) {
                __m512 a_vec = _mm512_loadu_ps(&A[k]);
                
                // Load BF16 values from current block and column
                XDNN_BF16 b_values[16];
                for (int i = 0; i < 16; i++) {
                    b_values[i] = blockB[(k + i) * ldb + j];
                }
                
                // Convert BF16 to FP32
                float b_float[16];
                for (int i = 0; i < 16; i++) {
                    b_float[i] = static_cast<float>(b_values[i]);
                }
                
                __m512 b_vec = _mm512_loadu_ps(b_float);
                
                // Multiply and accumulate
                __m512 prod = _mm512_mul_ps(a_vec, b_vec);
                sum += _mm512_reduce_add_ps(prod);
            }
            
            // Handle remaining elements
            for (; k < K; k++) {
                sum += A[k] * static_cast<float>(blockB[k * ldb + j]);
            }
#elif defined(__AVX2__)
            // AVX2 implementation
            int k = 0;
            
            // Process 8 elements at a time
            for (; k <= K - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[k]);
                
                // Load BF16 values from current block and column
                XDNN_BF16 b_values[8];
                for (int i = 0; i < 8; i++) {
                    b_values[i] = blockB[(k + i) * ldb + j];
                }
                
                // Convert BF16 to FP32
                float b_float[8];
                for (int i = 0; i < 8; i++) {
                    b_float[i] = static_cast<float>(b_values[i]);
                }
                
                __m256 b_vec = _mm256_loadu_ps(b_float);
                
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
                sum += A[k] * static_cast<float>(blockB[k * ldb + j]);
            }
#else
            // Scalar implementation
            for (int k = 0; k < K; k++) {
                sum += A[k] * static_cast<float>(blockB[k * ldb + j]);
            }
#endif
            
            // Store the result in the output
            C[outputOffset + j] = static_cast<XDNN_BF16>(sum);
        }
    }
}
