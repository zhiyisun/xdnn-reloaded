#include "sgemm_bf16bf16f32.h"
#include "intrinsic_ext.h"
#include <immintrin.h>
#include <cstring>

// Helper function: AVX horizontal sum (reduction)
float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    __m128 high128 = _mm256_extractf128_ps(x, 1);
    __m128 low128 = _mm256_castps256_ps128(x);
    __m128 add128 = _mm_add_ps(high128, low128);
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    __m128 move = _mm_movehl_ps(add128, add128);
    __m128 sum = _mm_add_ps(add128, move);
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    __m128 shuffle = _mm_shuffle_ps(sum, sum, 1);
    __m128 result = _mm_add_ss(sum, shuffle);
    return _mm_cvtss_f32(result);
}

/**
 * Implementation for standard small_sgemm_bf16bf16f32 function.
 * This function performs matrix multiplication: C = A * B
 * where:
 * - A is an M x K matrix in BF16 format
 * - B is a K x N matrix in BF16 format (if transB=false) or N x K matrix (if transB=true)
 * - C is an M x N matrix in F32 format
 * 
 * As specified in the header, this function assumes:
 * - M = 1 (single output row)
 * - transB = true (B is in column-major order)
 */
void small_sgemm_bf16bf16f32(bool transB, int M, int N, int K, 
                            const XDNN_BF16 *A, int lda, 
                            const XDNN_BF16 *B, int ldb, 
                            float *C, int ldc) {
    // Verify constraints from header comment
    if (M != 1 || !transB) {
        // In production code, we might want to handle this error differently
        return;
    }
    
    // Process each column of the output
    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        
        // Use AVX/AVX2 if available and K is large enough
        if (K >= 16) {
            int k = 0;
            
            // Process 8 elements at a time using AVX
            __m256 sum_vec = _mm256_setzero_ps();
            for (; k + 8 <= K; k += 8) {
                // Load 8 BF16 elements from A and convert to float
                __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(A + k)));
                
                // Load 8 BF16 elements from B and convert to float
                // (column-major since transB=true)
                __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(B + n * ldb + k)));
                
                // Multiply and accumulate
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            
            // Horizontal sum of the vector
            sum += _mm256_reduce_add_ps(sum_vec);
            
            // Process remaining elements
            for (; k < K; ++k) {
                float a_val = static_cast<float>(A[k]);
                float b_val = static_cast<float>(B[n * ldb + k]);
                sum += a_val * b_val;
            }
        } else {
            // Scalar path for small K
            for (int k = 0; k < K; ++k) {
                float a_val = static_cast<float>(A[k]);
                float b_val = static_cast<float>(B[n * ldb + k]);
                sum += a_val * b_val;
            }
        }
        
        // Store the result
        C[n] = sum;
    }
}

/**
 * Implementation for paged attention version of small_sgemm_bf16bf16f32_b.
 * This function performs matrix multiplication where B is organized in blocks:
 * C = A * B
 * where:
 * - A is an M x K matrix in BF16 format
 * - B is organized in blocks with strides defined by blockStride and blockSize
 * - C is an M x N matrix in F32 format
 * 
 * As specified in the header, this function assumes:
 * - M = 1 (single output row)
 * - transB = true (each block of B is in column-major order)
 */
void small_sgemm_bf16bf16f32_b(bool transB, int M, int N, int K, 
                              const XDNN_BF16 *A, int lda, 
                              const XDNN_BF16 *B, int ldb, 
                              float *C, int ldc, 
                              int *blockIndices, int blockStride, int blockSize) {
    // Verify constraints from header comment
    if (M != 1 || !transB) {
        // In production code, we might want to handle this error differently
        return;
    }
    
    // Reset output to zeros
    std::memset(C, 0, N * sizeof(float));
    
    // Calculate number of blocks
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Process each block
    for (int b = 0; b < numBlocks; b++) {
        // Get the block index from indices array
        int blockIdx = blockIndices[b];
        
        // Determine start position in the output
        int outputOffset = b * blockSize;
        
        // Determine start position in B based on block index
        const XDNN_BF16 *blockB = B + blockIdx * blockStride;
        
        // Process each column in this block (but not exceeding N)
        int columnsInThisBlock = std::min(blockSize, N - outputOffset);
        
        for (int j = 0; j < columnsInThisBlock; j++) {
            float sum = 0.0f;
            
            // Match the reference implementation exactly for correctness
            for (int k = 0; k < K; ++k) {
                float a_val = static_cast<float>(A[k]);
                float b_val = static_cast<float>(blockB[j * ldb + k]);
                sum += a_val * b_val;
            }
            
            // Store the result in the output
            C[outputOffset + j] = sum;
        }
    }
}
