#include "conversion.h"
#include "sgemm_f32nf4f32.h"
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include <cmath>

// Helper functions for activation
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

inline float gelu(float x) {
    // GELU approximation
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

// Helper function to convert NF4x2 to fp32 with table lookup
inline void nf4x2_to_fp32(const XDNN_NF4x2& nf4x2, float& val1, float& val2) {
    uint8_t idx1 = nf4x2.get_v1();
    uint8_t idx2 = nf4x2.get_v2();
    val1 = XDNN_NORMAL_FLOAT32[idx1];
    val2 = XDNN_NORMAL_FLOAT32[idx2];
}

// Quantize FP32 to NF4
void xdnn_sgemm_f32nf4f32_quantize(bool transB, int N, int K, const float *B, int ldb,
                                  float quantization_rate, XDNN_NF4x2 *quantizedB, int ldqb, float *scaleB, float *zeroB) {
    // Quantize B to NF4 with per-column scaling
    // NF4x2 means each byte contains two 4-bit indices into the normal float table
    
    if (!transB) {  // B is K x N
        for (int n = 0; n < N; n++) {
            // Find min and max value in the column
            float min_val = B[0 * ldb + n];
            float max_val = min_val;
            
            for (int k = 1; k < K; k++) {
                float val = B[k * ldb + n];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
            
            // Compute scale based on the max absolute value
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            if (quantization_rate < 1.0f) {
                // Apply quantization_rate to limit the range
                abs_max *= quantization_rate;
            }
            scaleB[n] = abs_max;
            zeroB[n] = 0.0f;  // Zero-point for symmetric quantization
            
            // Quantize the column
            float inv_scale = (abs_max > 0) ? 1.0f / abs_max : 0.0f;
            
            // Process two values at a time to create NF4x2
            for (int k = 0; k < K; k += 2) {
                float val1 = B[k * ldb + n] * inv_scale;
                float val2 = (k + 1 < K) ? B[(k + 1) * ldb + n] * inv_scale : 0.0f;
                
                // Clamp to [-1, 1]
                val1 = std::max(std::min(val1, 1.0f), -1.0f);
                val2 = std::max(std::min(val2, 1.0f), -1.0f);
                
                // Find closest index in the NF4 table
                uint8_t idx1 = 0;
                uint8_t idx2 = 0;
                float min_diff1 = std::abs(val1 - XDNN_NORMAL_FLOAT32[0]);
                float min_diff2 = std::abs(val2 - XDNN_NORMAL_FLOAT32[0]);
                
                for (uint8_t i = 1; i < 16; ++i) {
                    float diff1 = std::abs(val1 - XDNN_NORMAL_FLOAT32[i]);
                    if (diff1 < min_diff1) {
                        min_diff1 = diff1;
                        idx1 = i;
                    }
                    
                    float diff2 = std::abs(val2 - XDNN_NORMAL_FLOAT32[i]);
                    if (diff2 < min_diff2) {
                        min_diff2 = diff2;
                        idx2 = i;
                    }
                }
                
                // Store indices as NF4x2
                quantizedB[(k/2) * ldqb + n] = XDNN_NF4x2(idx1, idx2);
            }
        }
    } else {  // B is N x K
        for (int n = 0; n < N; n++) {
            // Find min and max value in the row
            float min_val = B[n * ldb + 0];
            float max_val = min_val;
            
            for (int k = 1; k < K; k++) {
                float val = B[n * ldb + k];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
            
            // Compute scale based on the max absolute value
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            if (quantization_rate < 1.0f) {
                // Apply quantization_rate to limit the range
                abs_max *= quantization_rate;
            }
            scaleB[n] = abs_max;
            zeroB[n] = 0.0f;  // Zero-point for symmetric quantization
            
            // Quantize the row
            float inv_scale = (abs_max > 0) ? 1.0f / abs_max : 0.0f;
            
            // Process two values at a time to create NF4x2
            for (int k = 0; k < K; k += 2) {
                float val1 = B[n * ldb + k] * inv_scale;
                float val2 = (k + 1 < K) ? B[n * ldb + k + 1] * inv_scale : 0.0f;
                
                // Clamp to [-1, 1]
                val1 = std::max(std::min(val1, 1.0f), -1.0f);
                val2 = std::max(std::min(val2, 1.0f), -1.0f);
                
                // Find closest index in the NF4 table
                uint8_t idx1 = 0;
                uint8_t idx2 = 0;
                float min_diff1 = std::abs(val1 - XDNN_NORMAL_FLOAT32[0]);
                float min_diff2 = std::abs(val2 - XDNN_NORMAL_FLOAT32[0]);
                
                for (uint8_t i = 1; i < 16; ++i) {
                    float diff1 = std::abs(val1 - XDNN_NORMAL_FLOAT32[i]);
                    if (diff1 < min_diff1) {
                        min_diff1 = diff1;
                        idx1 = i;
                    }
                    
                    float diff2 = std::abs(val2 - XDNN_NORMAL_FLOAT32[i]);
                    if (diff2 < min_diff2) {
                        min_diff2 = diff2;
                        idx2 = i;
                    }
                }
                
                // Store indices as NF4x2
                quantizedB[n * ldqb + k/2] = XDNN_NF4x2(idx1, idx2);
            }
        }
    }
}

// SGEMM implementation with NF4 inputs
void xdnn_sgemm_f32nf4f32(bool transA, bool transB, int M, int N, int K,
                         float alpha, const float *A, int lda, const XDNN_NF4x2 *B, int ldb, const float *scaleB, const float *zeroB,
                         float beta, float *C, int ldc) {
    // Apply beta scaling to C
    if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    // Matrix multiplication with alpha scaling and denormalization
    // K is the original K, but actual packed K is K/2 rounded up (since each NF4x2 contains two values)
    int packed_K = (K + 1) / 2;
    
    if (!transA && !transB) {
        // A: M×K, B: K×N
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < packed_K; k++) {
                    float val1, val2;
                    nf4x2_to_fp32(B[k * ldb + j], val1, val2);
                    
                    // Dequantize
                    val1 = val1 * scaleB[j] + zeroB[j];
                    val2 = val2 * scaleB[j] + zeroB[j];
                    
                    // Multiply and accumulate, handling odd K
                    int original_k = k * 2;
                    sum += A[i * lda + original_k] * val1;
                    if (original_k + 1 < K) {
                        sum += A[i * lda + original_k + 1] * val2;
                    }
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else if (transA && !transB) {
        // A: K×M, B: K×N
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < packed_K; k++) {
                    float val1, val2;
                    nf4x2_to_fp32(B[k * ldb + j], val1, val2);
                    
                    // Dequantize
                    val1 = val1 * scaleB[j] + zeroB[j];
                    val2 = val2 * scaleB[j] + zeroB[j];
                    
                    // Multiply and accumulate, handling odd K
                    int original_k = k * 2;
                    sum += A[original_k * lda + i] * val1;
                    if (original_k + 1 < K) {
                        sum += A[(original_k + 1) * lda + i] * val2;
                    }
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else if (!transA && transB) {
        // A: M×K, B: N×K
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < packed_K; k++) {
                    float val1, val2;
                    nf4x2_to_fp32(B[j * ldb + k], val1, val2);
                    
                    // Dequantize
                    val1 = val1 * scaleB[j] + zeroB[j];
                    val2 = val2 * scaleB[j] + zeroB[j];
                    
                    // Multiply and accumulate, handling odd K
                    int original_k = k * 2;
                    sum += A[i * lda + original_k] * val1;
                    if (original_k + 1 < K) {
                        sum += A[i * lda + original_k + 1] * val2;
                    }
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else { // transA && transB
        // A: K×M, B: N×K
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < packed_K; k++) {
                    float val1, val2;
                    nf4x2_to_fp32(B[j * ldb + k], val1, val2);
                    
                    // Dequantize
                    val1 = val1 * scaleB[j] + zeroB[j];
                    val2 = val2 * scaleB[j] + zeroB[j];
                    
                    // Multiply and accumulate, handling odd K
                    int original_k = k * 2;
                    sum += A[original_k * lda + i] * val1;
                    if (original_k + 1 < K) {
                        sum += A[(original_k + 1) * lda + i] * val2;
                    }
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// Pack matrix B for optimized computation
void xdnn_sgemm_f32nf4f32_packb(bool transB, int N, int K, const XDNN_NF4x2 *B, int ldb, XDNN_NF4x2 *packedB) {
    // Packing B for better cache locality in subsequent computations
    int packed_K = (K + 1) / 2;
    
    if (!transB) {
        // B is K×N
        for (int k = 0; k < packed_K; k++) {
            for (int n = 0; n < N; n++) {
                packedB[k * N + n] = B[k * ldb + n];
            }
        }
    } else {
        // B is N×K
        for (int k = 0; k < packed_K; k++) {
            for (int n = 0; n < N; n++) {
                packedB[k * N + n] = B[n * ldb + k];
            }
        }
    }
}

// Remaining functions for NF4 operations follow the same pattern as INT8 implementation
// with adjustments for the NF4 data format. These include:

// Compute SGEMM with pre-packed NF4 B matrix
void xdnn_sgemm_f32nf4f32_compute(bool transA, int M, int N, int K,
                                 float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
                                 float beta, float *C, int ldc) {
    // Apply beta scaling to C
    if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    // Matrix multiplication with pre-packed B
    int packed_K = (K + 1) / 2;
    
    if (!transA) {
        // A: M×K
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < packed_K; k++) {
                    float val1, val2;
                    nf4x2_to_fp32(packedB[k * N + j], val1, val2);
                    
                    // Dequantize
                    val1 = val1 * scaleB[j] + zeroB[j];
                    val2 = val2 * scaleB[j] + zeroB[j];
                    
                    // Multiply and accumulate, handling odd K
                    int original_k = k * 2;
                    sum += A[i * lda + original_k] * val1;
                    if (original_k + 1 < K) {
                        sum += A[i * lda + original_k + 1] * val2;
                    }
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else {
        // A: K×M
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < packed_K; k++) {
                    float val1, val2;
                    nf4x2_to_fp32(packedB[k * N + j], val1, val2);
                    
                    // Dequantize
                    val1 = val1 * scaleB[j] + zeroB[j];
                    val2 = val2 * scaleB[j] + zeroB[j];
                    
                    // Multiply and accumulate, handling odd K
                    int original_k = k * 2;
                    sum += A[original_k * lda + i] * val1;
                    if (original_k + 1 < K) {
                        sum += A[(original_k + 1) * lda + i] * val2;
                    }
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// Implementations of activation and bias functions follow the same pattern as the INT8 versions
// with adjustments for the NF4 data format

// Compute SGEMM with SiLU activation
void xdnn_sgemm_f32nf4f32_compute_silu(bool transA, int M, int N, int K,
                                      float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
                                      float beta, float *C, int ldc) {
    // Compute regular SGEMM
    xdnn_sgemm_f32nf4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Apply SiLU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = silu(C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with GELU activation
void xdnn_sgemm_f32nf4f32_compute_gelu(bool transA, int M, int N, int K,
                                      float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
                                      float beta, float *C, int ldc) {
    // Compute regular SGEMM
    xdnn_sgemm_f32nf4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Apply GELU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = gelu(C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with bias addition
void xdnn_sgemm_f32nf4f32_compute_biasadd(bool transA, int M, int N, int K,
                                         float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
                                         float beta, float *C, int ldc, const float *bias) {
    // Compute regular SGEMM
    xdnn_sgemm_f32nf4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Add bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += bias[j];
        }
    }
}

// Compute SGEMM with bias addition and ReLU activation
void xdnn_sgemm_f32nf4f32_compute_biasadd_relu(bool transA, int M, int N, int K,
                                         float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
                                         float beta, float *C, int ldc, const float *bias) {
    // Compute regular SGEMM with bias
    xdnn_sgemm_f32nf4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    
    // Apply ReLU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = std::max(0.0f, C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with bias and residential addition
void xdnn_sgemm_f32nf4f32_compute_residential(bool transA, int M, int N, int K,
                                           float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
                                           float beta, float *C, int ldc, const float *bias, const float *res, int ldres) {
    // Compute regular SGEMM
    xdnn_sgemm_f32nf4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Add bias and residual connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += bias[j] + res[i * ldres + j];
        }
    }
}

// Compute SGEMM with bias and scaled residential addition
void xdnn_sgemm_f32nf4f32_compute_resext(bool transA, int M, int N, int K,
                                      float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
                                      float beta, float *C, int ldc, const float *bias, 
                                      float gamma, const float *res, int ldres) {
    // Compute regular SGEMM
    xdnn_sgemm_f32nf4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Add bias and scaled residual connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += bias[j] + gamma * res[i * ldres + j];
        }
    }
}

// Compute SGEMM with element-wise multiplication of residual
void xdnn_sgemm_f32nf4f32_compute_resmul(bool transA, int M, int N, int K,
                                      float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
                                      float beta, float *C, int ldc, const float *res, int ldres) {
    // Compute regular SGEMM
    xdnn_sgemm_f32nf4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Perform element-wise multiplication with residual connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] *= res[i * ldres + j];
        }
    }
}

// Single-threaded optimized implementation for small matrices
void small_sgemm_f32nf4f32(int M, int N, int K, const float *A, int lda,
                         const XDNN_NF4x2 *B, int ldb, const float *scaleB, const float *zeroB, 
                         float *C, int ldc) {
    // Single-threaded optimized implementation for small matrix sizes
    // This is useful for situations where the overhead of threading would be too high
    
    int packed_K = (K + 1) / 2;  // Since each NF4x2 contains two values
    
    // Simple blocking for better cache usage
    constexpr int BLOCK_SIZE_M = 4;
    constexpr int BLOCK_SIZE_N = 16;
    constexpr int BLOCK_SIZE_K = 8;
    
    for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE_M) {
        int i_end = std::min(i0 + BLOCK_SIZE_M, M);
        
        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE_N) {
            int j_end = std::min(j0 + BLOCK_SIZE_N, N);
            
            for (int i = i0; i < i_end; i++) {
                for (int j = j0; j < j_end; j++) {
                    float sum = 0.0f;
                    
                    for (int kk = 0; kk < packed_K; kk += BLOCK_SIZE_K) {
                        int k_end = std::min(kk + BLOCK_SIZE_K, packed_K);
                        
                        for (int k = kk; k < k_end; k++) {
                            float val1, val2;
                            nf4x2_to_fp32(B[k * ldb + j], val1, val2);
                            
                            // Dequantize
                            val1 = val1 * scaleB[j] + zeroB[j];
                            val2 = val2 * scaleB[j] + zeroB[j];
                            
                            // Multiply and accumulate, handling odd K
                            int original_k = k * 2;
                            sum += A[i * lda + original_k] * val1;
                            if (original_k + 1 < K) {
                                sum += A[i * lda + original_k + 1] * val2;
                            }
                        }
                    }
                    
                    C[i * ldc + j] = sum;
                }
            }
        }
    }
}
