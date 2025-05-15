#include "conversion.h"
#include "sgemm_f32u4f32.h"
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

// Quantize FP32 to UINT4 (4-bit unsigned integer)
void xdnn_sgemm_f32u4f32_quantize(bool transB, int N, int K, const float *B, int ldb,
                                 float quantization_rate, XDNN_UINT4x2 *quantizedB, int ldqb, float *scaleB, float *zeroB) {
    // Quantize B to UINT4 with per-column scaling
    // UINT4x2 means each byte contains two 4-bit values
    
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
            
            // For unsigned quantization, we need to handle the zero-point carefully
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            if (quantization_rate < 1.0f) {
                // Apply quantization_rate to limit the range
                abs_max *= quantization_rate;
            }
            
            // For UINT4 (0-15 range), we set scale and zero point
            if (min_val < 0) {
                // We have negative values, need to shift them
                scaleB[n] = (max_val - min_val) / 15.0f;
                zeroB[n] = min_val;  // Zero-point is the min value
            } else {
                // All positive, simpler case
                scaleB[n] = max_val / 15.0f;
                zeroB[n] = 0.0f;  // Zero-point is 0
            }
            
            // Quantize the column, two values at a time to create UINT4x2
            for (int k = 0; k < K; k += 2) {
                float val1 = B[k * ldb + n];
                float val2 = (k + 1 < K) ? B[(k + 1) * ldb + n] : 0.0f;
                
                // Normalize to 0-15 range
                uint8_t q1 = static_cast<uint8_t>(std::round((val1 - zeroB[n]) / scaleB[n]));
                uint8_t q2 = static_cast<uint8_t>(std::round((val2 - zeroB[n]) / scaleB[n]));
                
                // Clamp to 0-15 range
                q1 = static_cast<uint8_t>(std::min(std::max(static_cast<unsigned int>(q1), 0u), 15u));
                q2 = static_cast<uint8_t>(std::min(std::max(static_cast<unsigned int>(q2), 0u), 15u));
                
                // Pack into UINT4x2
                quantizedB[(k/2) * ldqb + n] = XDNN_UINT4x2(q1, q2);
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
            
            // For unsigned quantization, handle zero-point carefully
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            if (quantization_rate < 1.0f) {
                // Apply quantization_rate to limit the range
                abs_max *= quantization_rate;
            }
            
            // For UINT4 (0-15 range), set scale and zero point
            if (min_val < 0) {
                // We have negative values, need to shift them
                scaleB[n] = (max_val - min_val) / 15.0f;
                zeroB[n] = min_val;  // Zero-point is the min value
            } else {
                // All positive, simpler case
                scaleB[n] = max_val / 15.0f;
                zeroB[n] = 0.0f;  // Zero-point is 0
            }
            
            // Quantize the row, two values at a time to create UINT4x2
            for (int k = 0; k < K; k += 2) {
                float val1 = B[n * ldb + k];
                float val2 = (k + 1 < K) ? B[n * ldb + k + 1] : 0.0f;
                
                // Normalize to 0-15 range
                uint8_t q1 = static_cast<uint8_t>(std::round((val1 - zeroB[n]) / scaleB[n]));
                uint8_t q2 = static_cast<uint8_t>(std::round((val2 - zeroB[n]) / scaleB[n]));
                
                // Clamp to 0-15 range
                q1 = static_cast<uint8_t>(std::min(std::max(static_cast<unsigned int>(q1), 0u), 15u));
                q2 = static_cast<uint8_t>(std::min(std::max(static_cast<unsigned int>(q2), 0u), 15u));
                
                // Pack into UINT4x2
                quantizedB[n * ldqb + k/2] = XDNN_UINT4x2(q1, q2);
            }
        }
    }
}

// Helper function to dequantize UINT4x2 to float
inline void uint4x2_to_float(const XDNN_UINT4x2& u4x2, float scale, float zero, float& val1, float& val2) {
    val1 = u4x2.get_v1() * scale + zero;
    val2 = u4x2.get_v2() * scale + zero;
}

// Main SGEMM implementation with UINT4 quantized B matrix
void xdnn_sgemm_f32u4f32(bool transA, bool transB, int M, int N, int K,
                        float alpha, const float *A, int lda, const XDNN_UINT4x2 *B, int ldb, const float *scaleB, const float *zeroB,
                        float beta, float *C, int ldc) {
    // Apply beta scaling to C
    if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    // Matrix multiplication with alpha scaling and dequantization
    // K is the original K, but actual packed K is (K+1)/2 (since each UINT4x2 contains two values)
    int packed_K = (K + 1) / 2;
    
    if (!transA && !transB) {
        // A: M×K, B: K×N
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int pk = 0; pk < packed_K; pk++) {
                    int k = pk * 2;
                    float b_val1, b_val2;
                    uint4x2_to_float(B[pk * ldb + j], scaleB[j], zeroB[j], b_val1, b_val2);
                    
                    sum += A[i * lda + k] * b_val1;
                    if (k + 1 < K) {
                        sum += A[i * lda + k + 1] * b_val2;
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
                for (int pk = 0; pk < packed_K; pk++) {
                    int k = pk * 2;
                    float b_val1, b_val2;
                    uint4x2_to_float(B[pk * ldb + j], scaleB[j], zeroB[j], b_val1, b_val2);
                    
                    sum += A[k * lda + i] * b_val1;
                    if (k + 1 < K) {
                        sum += A[(k + 1) * lda + i] * b_val2;
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
                for (int pk = 0; pk < packed_K; pk++) {
                    int k = pk * 2;
                    float b_val1, b_val2;
                    uint4x2_to_float(B[j * ldb + pk], scaleB[j], zeroB[j], b_val1, b_val2);
                    
                    sum += A[i * lda + k] * b_val1;
                    if (k + 1 < K) {
                        sum += A[i * lda + k + 1] * b_val2;
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
                for (int pk = 0; pk < packed_K; pk++) {
                    int k = pk * 2;
                    float b_val1, b_val2;
                    uint4x2_to_float(B[j * ldb + pk], scaleB[j], zeroB[j], b_val1, b_val2);
                    
                    sum += A[k * lda + i] * b_val1;
                    if (k + 1 < K) {
                        sum += A[(k + 1) * lda + i] * b_val2;
                    }
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// Pack matrix B for optimized computation
void xdnn_sgemm_f32u4f32_packb(bool transB, int N, int K, const XDNN_UINT4x2 *B, int ldb, XDNN_UINT4x2 *packedB) {
    // Packing B for better cache locality in subsequent computations
    int packed_K = (K + 1) / 2;
    
    if (!transB) {
        // B is packed_K×N
        for (int pk = 0; pk < packed_K; pk++) {
            for (int n = 0; n < N; n++) {
                packedB[pk * N + n] = B[pk * ldb + n];
            }
        }
    } else {
        // B is N×packed_K
        for (int pk = 0; pk < packed_K; pk++) {
            for (int n = 0; n < N; n++) {
                packedB[pk * N + n] = B[n * ldb + pk];
            }
        }
    }
}

// Compute SGEMM with pre-packed UINT4 B matrix
void xdnn_sgemm_f32u4f32_compute(bool transA, int M, int N, int K,
                                float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                float beta, float *C, int ldc) {
    // Apply beta scaling to C
    if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    // Matrix multiplication with pre-packed B and dequantization
    int packed_K = (K + 1) / 2;
    
    if (!transA) {
        // A: M×K
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int pk = 0; pk < packed_K; pk++) {
                    int k = pk * 2;
                    float b_val1, b_val2;
                    uint4x2_to_float(packedB[pk * N + j], scaleB[j], zeroB[j], b_val1, b_val2);
                    
                    sum += A[i * lda + k] * b_val1;
                    if (k + 1 < K) {
                        sum += A[i * lda + k + 1] * b_val2;
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
                for (int pk = 0; pk < packed_K; pk++) {
                    int k = pk * 2;
                    float b_val1, b_val2;
                    uint4x2_to_float(packedB[pk * N + j], scaleB[j], zeroB[j], b_val1, b_val2);
                    
                    sum += A[k * lda + i] * b_val1;
                    if (k + 1 < K) {
                        sum += A[(k + 1) * lda + i] * b_val2;
                    }
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// Compute SGEMM with SiLU activation
void xdnn_sgemm_f32u4f32_compute_silu(bool transA, int M, int N, int K,
                                     float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                     float beta, float *C, int ldc) {
    // Compute regular SGEMM
    xdnn_sgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Apply SiLU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = silu(C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with GELU activation
void xdnn_sgemm_f32u4f32_compute_gelu(bool transA, int M, int N, int K,
                                     float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                     float beta, float *C, int ldc) {
    // Compute regular SGEMM
    xdnn_sgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Apply GELU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = gelu(C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with bias addition
void xdnn_sgemm_f32u4f32_compute_biasadd(bool transA, int M, int N, int K,
                                        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                        float beta, float *C, int ldc, const float *bias) {
    // Compute regular SGEMM
    xdnn_sgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Add bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += bias[j];
        }
    }
}

// Compute SGEMM with bias addition and ReLU activation
void xdnn_sgemm_f32u4f32_compute_biasadd_relu(bool transA, int M, int N, int K,
                                             float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                             float beta, float *C, int ldc, const float *bias) {
    // Compute regular SGEMM with bias
    xdnn_sgemm_f32u4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    
    // Apply ReLU
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = std::max(0.0f, C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with residential connection
void xdnn_sgemm_f32u4f32_compute_residential(bool transA, int M, int N, int K,
                                            float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                            float beta, float *C, int ldc, const float *bias, const float *res, int ldres) {
    // Compute regular SGEMM with bias
    xdnn_sgemm_f32u4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    
    // Add residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += res[i * ldres + j];
        }
    }
}

// Compute SGEMM with extended residential connection
void xdnn_sgemm_f32u4f32_compute_resext(bool transA, int M, int N, int K,
                                       float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                       float beta, float *C, int ldc, const float *bias, 
                                       float gamma, const float *res, int ldres) {
    // Compute regular SGEMM with bias
    xdnn_sgemm_f32u4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    
    // Add scaled residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += gamma * res[i * ldres + j];
        }
    }
}

// Compute SGEMM with multiplicative residential connection
void xdnn_sgemm_f32u4f32_compute_resmul(bool transA, int M, int N, int K,
                                       float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                       float beta, float *C, int ldc, const float *res, int ldres) {
    // Compute regular SGEMM
    xdnn_sgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Multiply by residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] *= res[i * ldres + j];
        }
    }
}

// Small single-threaded GEMM implementation
void small_sgemm_f32u4f32(int M, int N, int K, const float *A, int lda,
                         const XDNN_UINT4x2 *B, int ldb, const float *scaleB, const float *zeroB, float *C, int ldc) {
    // Simple implementation optimized for small matrices
    int packed_K = (K + 1) / 2;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int pk = 0; pk < packed_K; pk++) {
                int k = pk * 2;
                float b_val1, b_val2;
                uint4x2_to_float(B[pk * ldb + j], scaleB[j], zeroB[j], b_val1, b_val2);
                
                sum += A[i * lda + k] * b_val1;
                if (k + 1 < K) {
                    sum += A[i * lda + k + 1] * b_val2;
                }
            }
            C[i * ldc + j] = sum;
        }
    }
}
