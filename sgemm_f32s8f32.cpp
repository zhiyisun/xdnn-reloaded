#include "conversion.h"
#include "sgemm_f32s8f32.h"
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

// Quantize FP32 to INT8
void xdnn_sgemm_f32s8f32_quantize(bool transB, int N, int K, const float *B, int ldb,
                                 float quantization_rate, int8_t *quantizedB, int ldqb, float *scaleB, float *zeroB) {
    // Quantize B to INT8 with per-column scaling
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
            scaleB[n] = abs_max / 127.0f;
            zeroB[n] = 0.0f;  // Zero-point for symmetric quantization
            
            // Quantize the column
            float inv_scale = (abs_max > 0) ? 127.0f / abs_max : 0.0f;
            for (int k = 0; k < K; k++) {
                float val = B[k * ldb + n];
                // Clamp to the range based on quantization_rate
                if (quantization_rate < 1.0f) {
                    val = std::max(std::min(val, abs_max), -abs_max);
                }
                quantizedB[k * ldqb + n] = static_cast<int8_t>(std::round(val * inv_scale));
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
            scaleB[n] = abs_max / 127.0f;
            zeroB[n] = 0.0f;  // Zero-point for symmetric quantization
            
            // Quantize the row
            float inv_scale = (abs_max > 0) ? 127.0f / abs_max : 0.0f;
            for (int k = 0; k < K; k++) {
                float val = B[n * ldb + k];
                // Clamp to the range based on quantization_rate
                if (quantization_rate < 1.0f) {
                    val = std::max(std::min(val, abs_max), -abs_max);
                }
                quantizedB[n * ldqb + k] = static_cast<int8_t>(std::round(val * inv_scale));
            }
        }
    }
}

// SGEMM implementation with INT8 inputs
void xdnn_sgemm_f32s8f32(bool transA, bool transB, int M, int N, int K,
                        float alpha, const float *A, int lda, const int8_t *B, int ldb, const float *scaleB, const float *zeroB,
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
    if (!transA && !transB) {
        // A: M×K, B: K×N
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * lda + k] * (B[k * ldb + j] * scaleB[j] + zeroB[j]);
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else if (transA && !transB) {
        // A: K×M, B: K×N
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[k * lda + i] * (B[k * ldb + j] * scaleB[j] + zeroB[j]);
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else if (!transA && transB) {
        // A: M×K, B: N×K
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * lda + k] * (B[j * ldb + k] * scaleB[j] + zeroB[j]);
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else { // transA && transB
        // A: K×M, B: N×K
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[k * lda + i] * (B[j * ldb + k] * scaleB[j] + zeroB[j]);
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// Pack matrix B for optimized computation
void xdnn_sgemm_f32s8f32_packb(bool transB, int N, int K, const int8_t *B, int ldb, int8_t *packedB) {
    // Packing B for better cache locality in subsequent computations
    if (!transB) {
        // B is K×N
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                packedB[k * N + n] = B[k * ldb + n];
            }
        }
    } else {
        // B is N×K
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                packedB[k * N + n] = B[n * ldb + k];
            }
        }
    }
}

// Compute SGEMM with pre-packed INT8 B matrix
void xdnn_sgemm_f32s8f32_compute(bool transA, int M, int N, int K,
                                float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
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
    if (!transA) {
        // A: M×K
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * lda + k] * (packedB[k * N + j] * scaleB[j] + zeroB[j]);
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else {
        // A: K×M
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[k * lda + i] * (packedB[k * N + j] * scaleB[j] + zeroB[j]);
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// Compute SGEMM with SiLU activation
void xdnn_sgemm_f32s8f32_compute_silu(bool transA, int M, int N, int K,
                                     float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                     float beta, float *C, int ldc) {
    // Compute regular SGEMM
    xdnn_sgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Apply SiLU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = silu(C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with GELU activation
void xdnn_sgemm_f32s8f32_compute_gelu(bool transA, int M, int N, int K,
                                     float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                     float beta, float *C, int ldc) {
    // Compute regular SGEMM
    xdnn_sgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Apply GELU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = gelu(C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with bias addition
void xdnn_sgemm_f32s8f32_compute_biasadd(bool transA, int M, int N, int K,
                                        float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                        float beta, float *C, int ldc, const float *bias) {
    // Compute regular SGEMM
    xdnn_sgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Add bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += bias[j];
        }
    }
}

// Compute SGEMM with bias addition and ReLU activation
void xdnn_sgemm_f32s8f32_compute_biasadd_relu(bool transA, int M, int N, int K,
                                             float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                             float beta, float *C, int ldc, const float *bias) {
    // Compute regular SGEMM with bias
    xdnn_sgemm_f32s8f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    
    // Apply ReLU
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = std::max(0.0f, C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with residential connection
void xdnn_sgemm_f32s8f32_compute_residential(bool transA, int M, int N, int K,
                                            float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                            float beta, float *C, int ldc, const float *bias, const float *res, int ldres) {
    // Compute regular SGEMM with bias
    xdnn_sgemm_f32s8f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    
    // Add residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += res[i * ldres + j];
        }
    }
}

// Compute SGEMM with extended residential connection
void xdnn_sgemm_f32s8f32_compute_resext(bool transA, int M, int N, int K,
                                       float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                       float beta, float *C, int ldc, const float *bias, 
                                       float gamma, const float *res, int ldres) {
    // Compute regular SGEMM with bias
    xdnn_sgemm_f32s8f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    
    // Add scaled residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += gamma * res[i * ldres + j];
        }
    }
}

// Compute SGEMM with multiplicative residential connection
void xdnn_sgemm_f32s8f32_compute_resmul(bool transA, int M, int N, int K,
                                       float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                       float beta, float *C, int ldc, const float *res, int ldres) {
    // Compute regular SGEMM
    xdnn_sgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Multiply by residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] *= res[i * ldres + j];
        }
    }
}

// Small single-threaded GEMM implementation
void small_sgemm_f32s8f32(int M, int N, int K, const float *A, int lda,
                         const int8_t *B, int ldb, const float *scaleB, const float *zeroB, float *C, int ldc) {
    // Simple implementation for small matrices
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * (B[k * ldb + j] * scaleB[j] + zeroB[j]);
            }
            C[i * ldc + j] = sum;
        }
    }
}
