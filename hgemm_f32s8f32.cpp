#include "conversion.h"
#include "hgemm_f32s8f32.h"
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

// HGEMM constants for int8 operations
#define HGEMM_MR 8
#define HGEMM_NR 8

// Helper function to compute SILU activation: x * sigmoid(x)
inline float silu_activate(float x) {
    return x / (1.0f + expf(-x));
}

// Helper function to compute GELU activation
inline float gelu_activate(float x) {
    // Approximation of GELU
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Quantize a matrix from float to int8
void xdnn_hgemm_f32s8f32_quantize(bool transB, int N, int K, const float *B, int ldb,
        float quantization_rate, int8_t *quantizedB, int ldqb, float *scaleB, float *zeroB) {
    
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
            float absMax = std::max(std::abs(maxVal), std::abs(minVal));
            
            // Apply quantization rate if specified
            if (quantization_rate > 0.0f && quantization_rate < 1.0f) {
                absMax *= quantization_rate;
            }
            
            // Avoid division by zero
            float scale = absMax > 0.0f ? absMax / 127.0f : 1.0f;
            scaleB[n] = scale;
            zeroB[n] = 0.0f;  // For symmetric quantization, zero point is 0
            
            // Quantize the column
            for (int k = 0; k < K; k++) {
                float val = B[n * ldb + k];
                val = std::min(std::max(val, -absMax), absMax);  // Clamp to range
                quantizedB[n * ldqb + k] = static_cast<int8_t>(std::round(val / scale));
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
            float absMax = std::max(std::abs(maxVal), std::abs(minVal));
            
            // Apply quantization rate if specified
            if (quantization_rate > 0.0f && quantization_rate < 1.0f) {
                absMax *= quantization_rate;
            }
            
            // Avoid division by zero
            float scale = absMax > 0.0f ? absMax / 127.0f : 1.0f;
            scaleB[n] = scale;
            zeroB[n] = 0.0f;  // For symmetric quantization, zero point is 0
            
            // Quantize the column
            for (int k = 0; k < K; k++) {
                float val = B[k * ldb + n];
                val = std::min(std::max(val, -absMax), absMax);  // Clamp to range
                quantizedB[k * ldqb + n] = static_cast<int8_t>(std::round(val / scale));
            }
        }
    }
}

// Main HGEMM function that handles different matrix layouts
void xdnn_hgemm_f32s8f32(bool transA, bool transB, int M, int N, int K,
        float alpha, const float *A, int lda, const int8_t *quantizedB, int ldb, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc) {
    
    // Allocate temporary packed B matrix
    size_t packedB_size = (size_t)((N + HGEMM_NR - 1) / HGEMM_NR) * 
                         ((K + HGEMM_KC - 1) / HGEMM_KC) * HGEMM_KC * HGEMM_NR * sizeof(int8_t);
    int8_t *packedB = new int8_t[packedB_size / sizeof(int8_t)];
    
    // Pack matrix B for better cache locality
    xdnn_hgemm_f32s8f32_packb(transB, N, K, quantizedB, ldb, packedB);
    
    // Compute the matrix multiplication
    xdnn_hgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Free temporary packed matrix
    delete[] packedB;
}

// Pack matrix B for efficient computation
void xdnn_hgemm_f32s8f32_packb(bool transB, int N, int K, const int8_t *quantizedB, int ldb, int8_t *packedB) {
    // Pack matrix B into block format for better cache usage during computation
    int nb = (N + HGEMM_NR - 1) / HGEMM_NR;
    int kb = (K + HGEMM_KC - 1) / HGEMM_KC;
    
    // Zero initialize the entire packed buffer
    memset(packedB, 0, nb * kb * HGEMM_KC * HGEMM_NR * sizeof(int8_t));
    
    for (int k_block = 0; k_block < K; k_block += HGEMM_KC) {
        int k_size = std::min(HGEMM_KC, K - k_block);
        
        for (int n_block = 0; n_block < N; n_block += HGEMM_NR) {
            int n_size = std::min(HGEMM_NR, N - n_block);
            
            // Calculate offset in the packed buffer
            int offset = (k_block / HGEMM_KC) * nb * HGEMM_KC * HGEMM_NR + 
                         (n_block / HGEMM_NR) * HGEMM_KC * HGEMM_NR;
            
            if (transB) {
                // B is in N x K format
                for (int k = 0; k < k_size; k++) {
                    for (int n = 0; n < n_size; n++) {
                        packedB[offset + k * HGEMM_NR + n] = quantizedB[(n_block + n) * ldb + (k_block + k)];
                    }
                }
            } else {
                // B is in K x N format
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
void xdnn_hgemm_f32s8f32_compute(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
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
    
    int nb = (N + HGEMM_NR - 1) / HGEMM_NR;
    
    // Main computation loop with blocking for cache efficiency
    for (int m = 0; m < M; m += HGEMM_MR) {
        int m_size = std::min(HGEMM_MR, M - m);
        
        for (int kb = 0; kb < K; kb += HGEMM_KC) {
            int k_size = std::min(HGEMM_KC, K - kb);
            
            for (int nb_idx = 0; nb_idx < nb; nb_idx++) {
                int n = nb_idx * HGEMM_NR;
                int n_size = std::min(HGEMM_NR, N - n);
                
                if (n_size <= 0) continue;
                
                // Calculate offset in the packed buffer
                int offset = (kb / HGEMM_KC) * nb * HGEMM_KC * HGEMM_NR + 
                             nb_idx * HGEMM_KC * HGEMM_NR;
                
                // Process one micro-kernel
                for (int i = 0; i < m_size; i++) {
                    for (int j = 0; j < n_size; j++) {
                        float sum = 0.0f;
                        
                        if (transA) {
                            for (int k = 0; k < k_size; k++) {
                                // Get scale factor based on groupsize or per-column
                                float scale = groupsize > 0 ? 
                                    scaleB[(n + j) / groupsize] : scaleB[n + j];
                                
                                sum += A[(kb + k) * lda + (m + i)] * 
                                      static_cast<float>(packedB[offset + k * HGEMM_NR + j]) * scale;
                            }
                        } else {
                            for (int k = 0; k < k_size; k++) {
                                // Get scale factor based on groupsize or per-column
                                float scale = groupsize > 0 ? 
                                    scaleB[(n + j) / groupsize] : scaleB[n + j];
                                
                                sum += A[(m + i) * lda + (kb + k)] * 
                                      static_cast<float>(packedB[offset + k * HGEMM_NR + j]) * scale;
                            }
                        }
                        
                        // Update C with accumulated result
                        C[(m + i) * ldc + (n + j)] += alpha * sum;
                    }
                }
            }
        }
    }
}

// HGEMM with SILU activation
void xdnn_hgemm_f32s8f32_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, int groupsize) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, groupsize);
    
    // Then apply SILU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = silu_activate(C[i * ldc + j]);
        }
    }
}

// HGEMM with GELU activation
void xdnn_hgemm_f32s8f32_compute_gelu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, int groupsize) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, groupsize);
    
    // Then apply GELU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = gelu_activate(C[i * ldc + j]);
        }
    }
}

// HGEMM with extended residual connection
void xdnn_hgemm_f32s8f32_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias,
        float gamma, const float *res, int ldres, int groupsize) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, groupsize);
    
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
void xdnn_hgemm_f32s8f32_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *res, int ldres, int groupsize) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, groupsize);
    
    // Then multiply by residual (element-wise)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] *= res[i * ldres + j];
        }
    }
}

// HGEMM with bias addition
void xdnn_hgemm_f32s8f32_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias, int groupsize) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, groupsize);
    
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
void xdnn_hgemm_f32s8f32_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias, int groupsize) {
    
    // First compute HGEMM with bias addition
    xdnn_hgemm_f32s8f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, groupsize);
    
    // Then apply ReLU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = std::max(0.0f, C[i * ldc + j]);
        }
    }
}

// HGEMM with bias addition and residual connection
void xdnn_hgemm_f32s8f32_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias, const float *res, int ldres, int groupsize) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, groupsize);
    
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
void small_hgemm_f32s8f32(int M, int N, int K, const float *A, int lda,
        const int8_t *quantizedB, int ldb, const float *scaleB, const float *zeroB, float *C, int ldc) {
    
    // First zero initialize output
    for (int i = 0; i < M; i++) {
        memset(&C[i * ldc], 0, N * sizeof(float));
    }
    
    // Basic implementation for small matrices without blocking
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * static_cast<float>(quantizedB[k * ldb + j]) * scaleB[j];
            }
            
            C[i * ldc + j] = sum;
        }
    }
}
