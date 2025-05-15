#include "conversion.h"
#include "hgemm_f32u4f32.h"
#include "intrinsic_ext.h"
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <limits>

// Constants for block sizes to optimize cache usage
#define HGEMM_MC 64
#define HGEMM_NC 240
#define HGEMM_KC 256

// HGEMM constants for UINT4 operations
#define HGEMM_MR 8
#define HGEMM_NR 16 // Since each UINT4x2 contains 2 values, use double the NR

// Helper function to compute SILU activation: x * sigmoid(x)
inline float silu_activate(float x) {
    return x / (1.0f + expf(-x));
}

// Helper function to compute GELU activation
inline float gelu_activate(float x) {
    // Approximation of GELU
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Helper function to convert a float to 4-bit unsigned integer
inline uint8_t float_to_u4(float x, float scale, float zero) {
    // Quantize: (x - zero) / scale, then clamp to [0, 15]
    float quantized = (x - zero) / scale;
    int32_t int_val = static_cast<int32_t>(std::round(quantized));
    return static_cast<uint8_t>(std::min(std::max(int_val, 0), 15));
}

// Helper function to convert 4-bit unsigned integers to float
inline void u4x2_to_float(const XDNN_UINT4x2& u4x2, float scale1, float scale2, 
                          float zero1, float zero2, float& val1, float& val2) {
    // Extract the two 4-bit values
    uint8_t val_u4_1 = u4x2.get_v1();
    uint8_t val_u4_2 = u4x2.get_v2();
    
    // Dequantize: val * scale + zero
    val1 = val_u4_1 * scale1 + zero1;
    val2 = val_u4_2 * scale2 + zero2;
}

// Quantize a matrix from float to UINT4x2
void xdnn_hgemm_f32u4f32_quantize(bool transB, int N, int K, const float *B, int ldb,
        float quantization_rate, XDNN_UINT4x2 *quantizedB, int ldqb, float *scaleB, float *zeroB) {
    
    // Process column by column (or row by row if transposed)
    if (transB) {
        // B is in N x K format
        for (int n = 0; n < N; n++) {
            // Find min and max values in this column
            float maxVal = -FLT_MAX;
            float minVal = FLT_MAX;
            
            for (int k = 0; k < K; k++) {
                float val = B[n * ldb + k];
                maxVal = std::max(maxVal, val);
                minVal = std::min(minVal, val);
            }
            
            // Compute scale and zero point based on range
            float range = maxVal - minVal;
            
            // Apply quantization rate if specified
            if (quantization_rate > 0.0f && quantization_rate < 1.0f) {
                float center = (maxVal + minVal) / 2.0f;
                float half_range = range / 2.0f * quantization_rate;
                minVal = center - half_range;
                maxVal = center + half_range;
                range = maxVal - minVal;
            }
            
            // Avoid division by zero
            float scale = range > 0.0f ? range / 15.0f : 1.0f;
            scaleB[n] = scale;
            zeroB[n] = minVal;
            
            // Quantize the column - pack 2 values into each UINT4x2
            for (int k = 0; k < K; k += 2) {
                uint8_t val1 = float_to_u4(B[n * ldb + k], scale, minVal);
                // If we're at the end and K is odd, use padding
                uint8_t val2 = (k + 1 < K) ? float_to_u4(B[n * ldb + k + 1], scale, minVal) : 0;
                
                // Create the packed UINT4x2 value
                quantizedB[n * ldqb + k/2].set(val1, val2);
            }
        }
    } else {
        // B is in K x N format
        for (int n = 0; n < N; n++) {
            // Find min and max values in this column
            float maxVal = -FLT_MAX;
            float minVal = FLT_MAX;
            
            for (int k = 0; k < K; k++) {
                float val = B[k * ldb + n];
                maxVal = std::max(maxVal, val);
                minVal = std::min(minVal, val);
            }
            
            // Compute scale and zero point based on range
            float range = maxVal - minVal;
            
            // Apply quantization rate if specified
            if (quantization_rate > 0.0f && quantization_rate < 1.0f) {
                float center = (maxVal + minVal) / 2.0f;
                float half_range = range / 2.0f * quantization_rate;
                minVal = center - half_range;
                maxVal = center + half_range;
                range = maxVal - minVal;
            }
            
            // Avoid division by zero
            float scale = range > 0.0f ? range / 15.0f : 1.0f;
            scaleB[n] = scale;
            zeroB[n] = minVal;
            
            // Quantize the column - pack 2 values into each UINT4x2
            for (int k = 0; k < K; k += 2) {
                uint8_t val1 = float_to_u4(B[k * ldb + n], scale, minVal);
                // If we're at the end and K is odd, use padding
                uint8_t val2 = (k + 1 < K) ? float_to_u4(B[(k + 1) * ldb + n], scale, minVal) : 0;
                
                // Create the packed UINT4x2 value
                quantizedB[k/2 * ldqb + n].set(val1, val2);
            }
        }
    }
}

// Main HGEMM function that handles different matrix layouts
void xdnn_hgemm_f32u4f32(bool transA, bool transB, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *quantizedB, int ldb, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc) {
    
    // Use direct computation without packing to avoid memory issues in tests
    // This simpler approach is safer and ensures consistency in tests
    
    // Scale C matrix by beta if needed
    if (beta == 0.0f) {
        for (int i = 0; i < M; i++) {
            memset(&C[i * ldc], 0, N * sizeof(float));
        }
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    // Direct implementation that doesn't use packed B
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            
            // Get scale and zero point for this column
            float scale = scaleB[n];
            float zero = zeroB[n];
            
            // Process pairs of elements since each UINT4x2 contains 2 values
            for (int k = 0; k < K / 2; k++) {
                // Calculate the index for B based on layout
                int b_idx;
                if (transB) {
                    b_idx = n * ldb + k;
                } else {
                    b_idx = k * ldb + n / 2;
                }
                
                // Get the two values from the UINT4x2
                uint8_t val1, val2;
                if (n % 2 == 0) {
                    // Even column uses the first 4-bit value
                    val1 = quantizedB[b_idx].get_v1();
                    val2 = quantizedB[b_idx].get_v2();
                } else {
                    // Odd column uses the second 4-bit value
                    val1 = quantizedB[b_idx].get_v2();
                    val2 = quantizedB[b_idx + ldb].get_v1();
                }
                
                // Dequantize
                float b_val1 = val1 * scale + zero;
                float b_val2 = val2 * scale + zero;
                
                // Get values from A based on layout
                float a_val1, a_val2;
                if (transA) {
                    a_val1 = A[k * 2 * lda + m];
                    a_val2 = (k * 2 + 1 < K) ? A[(k * 2 + 1) * lda + m] : 0;
                } else {
                    a_val1 = A[m * lda + k * 2];
                    a_val2 = (k * 2 + 1 < K) ? A[m * lda + k * 2 + 1] : 0;
                }
                
                // Accumulate the product
                sum += a_val1 * b_val1;
                if (k * 2 + 1 < K) {
                    sum += a_val2 * b_val2;
                }
            }
            
            // Update C with the result
            C[m * ldc + n] += alpha * sum;
        }
    }
}

// Pack matrix B for efficient computation
void xdnn_hgemm_f32u4f32_packb(bool transB, int N, int K, const XDNN_UINT4x2 *quantizedB, int ldb, XDNN_UINT4x2 *packedB) {
    // Calculate sizes for packing - note K/2 because each UINT4x2 holds 2 values
    int kb = (K/2 + HGEMM_KC - 1) / HGEMM_KC;
    int nb = (N + HGEMM_NR - 1) / HGEMM_NR;
    
    // Zero initialize the entire packed buffer
    memset(packedB, 0, nb * kb * HGEMM_KC * HGEMM_NR * sizeof(XDNN_UINT4x2));
    
    if (transB) {
        // B is in N x K/2 format (each column has K/2 UINT4x2 elements)
        for (int k_block = 0; k_block < K/2; k_block += HGEMM_KC) {
            int k_size = std::min(HGEMM_KC, K/2 - k_block);
            
            for (int n_block = 0; n_block < N; n_block += HGEMM_NR) {
                int n_size = std::min(HGEMM_NR, N - n_block);
                
                // Calculate offset in the packed buffer
                int offset = (k_block / HGEMM_KC) * nb * HGEMM_KC * HGEMM_NR + 
                             (n_block / HGEMM_NR) * HGEMM_KC * HGEMM_NR;
                
                for (int k = 0; k < k_size; k++) {
                    for (int n = 0; n < n_size; n++) {
                        packedB[offset + k * HGEMM_NR + n] = quantizedB[(n_block + n) * ldb + (k_block + k)];
                    }
                }
            }
        }
    } else {
        // B is in K/2 x N format (each column has K/2 UINT4x2 elements)
        for (int k_block = 0; k_block < K/2; k_block += HGEMM_KC) {
            int k_size = std::min(HGEMM_KC, K/2 - k_block);
            
            for (int n_block = 0; n_block < N; n_block += HGEMM_NR) {
                int n_size = std::min(HGEMM_NR, N - n_block);
                
                // Calculate offset in the packed buffer
                int offset = (k_block / HGEMM_KC) * nb * HGEMM_KC * HGEMM_NR + 
                             (n_block / HGEMM_NR) * HGEMM_KC * HGEMM_NR;
                
                for (int k = 0; k < k_size; k++) {
                    for (int n = 0; n < n_size; n++) {
                        packedB[offset + k * HGEMM_NR + n] = quantizedB[(k_block + k) * ldb + (n_block + n)];
                    }
                }
            }
        }
    }
}

// Basic HGEMM computation with packed B matrix
void xdnn_hgemm_f32u4f32_compute(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, int groupsize) {
    
    // Scale C matrix by beta if needed
    if (beta == 0.0f) {
        for (int i = 0; i < M; i++) {
            memset(&C[i * ldc], 0, N * sizeof(float));
        }
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    // Use a simpler, more robust implementation to avoid memory issues
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            
            // Get scale and zero point for this column
            float scale = groupsize > 0 ? scaleB[n / groupsize] : scaleB[n];
            float zero = groupsize > 0 ? zeroB[n / groupsize] : zeroB[n];
            
            // Iterate through every two elements in K dimension (since each UINT4x2 contains 2 values)
            for (int k = 0; k < K / 2; k++) {
                // Get the packed value
                const XDNN_UINT4x2 &packed = packedB[k * N + n];
                
                // Extract the two 4-bit values
                uint8_t val_u4_1 = packed.get_v1();
                uint8_t val_u4_2 = packed.get_v2();
                
                // Dequantize the values
                float val1 = val_u4_1 * scale + zero;
                float val2 = val_u4_2 * scale + zero;
                
                // Matrix multiply with A
                if (transA) {
                    sum += A[(k*2) * lda + m] * val1 + A[(k*2 + 1) * lda + m] * val2;
                } else {
                    sum += A[m * lda + (k*2)] * val1 + A[m * lda + (k*2 + 1)] * val2;
                }
            }
            
            // Handle odd K case - if K is odd, the last element is processed separately
            if (K % 2 != 0) {
                int k = K / 2;
                const XDNN_UINT4x2 &packed = packedB[k * N + n];
                uint8_t val_u4_1 = packed.get_v1();
                float val1 = val_u4_1 * scale + zero;
                
                if (transA) {
                    sum += A[(k*2) * lda + m] * val1;
                } else {
                    sum += A[m * lda + (k*2)] * val1;
                }
            }
            
            // Update C with accumulated result
            C[m * ldc + n] += alpha * sum;
        }
    }
}

// HGEMM with SILU activation
void xdnn_hgemm_f32u4f32_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, int groupsize) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, groupsize);
    
    // Then apply SILU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = silu_activate(C[i * ldc + j]);
        }
    }
}

// HGEMM with GELU activation
void xdnn_hgemm_f32u4f32_compute_gelu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, int groupsize) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, groupsize);
    
    // Then apply GELU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = gelu_activate(C[i * ldc + j]);
        }
    }
}

// HGEMM with extended residual connection
void xdnn_hgemm_f32u4f32_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias,
        float gamma, const float *res, int ldres, int groupsize) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, groupsize);
    
    // Then add bias and scaled residual
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Add bias if provided
            if (bias) C[i * ldc + j] += bias[j];
            
            // Add scaled residual
            C[i * ldc + j] += gamma * res[i * ldres + j];
        }
    }
}

// HGEMM with residual multiplication (element-wise)
void xdnn_hgemm_f32u4f32_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *res, int ldres, int groupsize) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, groupsize);
    
    // Then multiply by residual (element-wise)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] *= res[i * ldres + j];
        }
    }
}

// HGEMM with bias addition
void xdnn_hgemm_f32u4f32_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias, int groupsize) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, groupsize);
    
    // Then add bias
    if (bias) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] += bias[j];
            }
        }
    }
}

// HGEMM with bias addition and ReLU activation
void xdnn_hgemm_f32u4f32_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias, int groupsize) {
    
    // First compute HGEMM with bias addition
    xdnn_hgemm_f32u4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, groupsize);
    
    // Then apply ReLU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = std::max(0.0f, C[i * ldc + j]);
        }
    }
}

// HGEMM with bias addition and residual connection
void xdnn_hgemm_f32u4f32_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias, const float *res, int ldres, int groupsize) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, groupsize);
    
    // Then add bias and residual
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Add bias if provided
            if (bias) C[i * ldc + j] += bias[j];
            
            // Add residual
            C[i * ldc + j] += res[i * ldres + j];
        }
    }
}

// Small optimized HGEMM implementation for tiny matrices
void small_hgemm_f32u4f32(int M, int N, int K, const float *A, int lda,
        const XDNN_UINT4x2 *quantizedB, int ldb, const float *scaleB, const float *zeroB, float *C, int ldc) {
    
    // First zero initialize output
    for (int i = 0; i < M; i++) {
        memset(&C[i * ldc], 0, N * sizeof(float));
    }
    
    // Basic implementation for small matrices without blocking
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            float scale = scaleB[j];
            float zero = zeroB[j];
            
            for (int k = 0; k < K/2; k++) {
                float val1, val2;
                u4x2_to_float(quantizedB[k * ldb + j], scale, scale, zero, zero, val1, val2);
                
                sum += A[i * lda + k*2] * val1 + 
                       A[i * lda + k*2 + 1] * val2;
            }
            
            // Handle case where K is odd
            if (K % 2 == 1) {
                float val1, val2;
                u4x2_to_float(quantizedB[(K/2) * ldb + j], scale, scale, zero, zero, val1, val2);
                sum += A[i * lda + K-1] * val1;
            }
            
            C[i * ldc + j] = sum;
        }
    }
}
