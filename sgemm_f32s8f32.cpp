#include "conversion.h"
#include "sgemm_f32s8f32.h"
#include "debug_print.h"
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include <cmath>
#include <limits>

// Helper functions for activation
inline float silu(float x) {
    DEBUG_PRINT();
    return x / (1.0f + std::exp(-x));
}

inline float gelu(float x) {
    DEBUG_PRINT();
    // GELU approximation
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

// Quantize FP32 to INT8 (8-bit signed integer)
void xdnn_sgemm_f32s8f32_quantize(bool transB, int N, int K, const float *B, int ldb,
                                 float quantization_rate, int8_t *quantizedB, int ldqb, float *scaleB, float *zeroB) {
    DEBUG_PRINT();
    const int num_quant_cols = transB ? K : N;
    const int num_quant_rows = transB ? N : K;

    if (num_quant_rows == 0 || num_quant_cols == 0) {
        // Zero out the output arrays if they're provided
        for (int j = 0; j < num_quant_cols; ++j) {
            scaleB[j] = 0.0f;
            zeroB[j] = 0.0f;
        }
        return;
    }
    
    // Calculate scale and zero point for each column
    for (int j = 0; j < num_quant_cols; ++j) {
        // Find min and max values in this column
        float col_min = std::numeric_limits<float>::max();
        float col_max = std::numeric_limits<float>::lowest();
        
        for (int i = 0; i < num_quant_rows; ++i) {
            float val = transB ? B[j * ldb + i] : B[i * ldb + j];
            col_min = std::min(col_min, val);
            col_max = std::max(col_max, val);
        }
        
        // Calculate midpoint of the range
        float midpoint = (col_max + col_min) / 2.0f;
        
        // Calculate absolute max as max distance from midpoint
        float abs_max = std::max(std::abs(col_max - midpoint), std::abs(col_min - midpoint));
        
        // Apply quantization_rate if needed
        if (quantization_rate > 0.0f && quantization_rate < 1.0f) {
            abs_max *= quantization_rate;
        }
        
        // Calculate scale for quantization
        float scale = (abs_max > 0) ? abs_max / 127.0f : 1e-9f;
        
        // Store scale and zero point
        scaleB[j] = scale;
        zeroB[j] = midpoint; // Use midpoint as zero point
        
        // Quantize values in this column
        for (int i = 0; i < num_quant_rows; ++i) {
            float original_val = transB ? B[j * ldb + i] : B[i * ldb + j];
            
            // Quantization with zero point: q = round((val - zero_point) / scale)
            float q = std::round((original_val - midpoint) / scale);
            
            // Clamp to int8 range [-127, 127]
            q = std::max(-127.0f, std::min(127.0f, q));
            
            // Store the result
            int output_idx = i * ldqb + j;
            quantizedB[output_idx] = static_cast<int8_t>(q);
        }
    }
}

// Main SGEMM implementation with INT8 quantized B matrix
void xdnn_sgemm_f32s8f32(bool transA, bool transB, int M, int N, int K,
                        float alpha, const float *A, int lda, const int8_t *B, int ldb, const float *scaleB, const float *zeroB,
                        float beta, float *C, int ldc) {
    DEBUG_PRINT();
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
                    // Dequantize B value
                    float b_val = static_cast<float>(B[k * ldb + j]) * scaleB[j] + zeroB[j];
                    sum += A[i * lda + k] * b_val;
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
                    // Dequantize B value
                    float b_val = static_cast<float>(B[k * ldb + j]) * scaleB[j] + zeroB[j];
                    sum += A[k * lda + i] * b_val;
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
                    // Dequantize B value
                    float b_val = static_cast<float>(B[j * ldb + k]) * scaleB[j] + zeroB[j];
                    sum += A[i * lda + k] * b_val;
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
                    // Dequantize B value
                    float b_val = static_cast<float>(B[j * ldb + k]) * scaleB[j] + zeroB[j];
                    sum += A[k * lda + i] * b_val;
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// Pack matrix B for optimized computation
void xdnn_sgemm_f32s8f32_packb(bool transB, int N, int K, const int8_t *B, int ldb, int8_t *packedB) {
    DEBUG_PRINT();
    // Output packedB is row-major KxN, tightly packed
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            int src_idx = transB ? (n * ldb + k) : (k * ldb + n);
            int dst_idx = k * N + n;
            packedB[dst_idx] = B[src_idx];
        }
    }
}

// Compute SGEMM with pre-packed INT8 B matrix
void xdnn_sgemm_f32s8f32_compute(bool transA, int M, int N, int K,
                                float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                float beta, float *C, int ldc) {
    DEBUG_PRINT();
    // Handle beta scaling
    if (beta == 0.0f) {
        // Zero out the output matrix
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                C[m * ldc + n] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        // Scale the output matrix by beta
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                C[m * ldc + n] *= beta;
            }
        }
    }
    
    // Skip computation if dimensions are zero
    if (M == 0 || N == 0 || K == 0) return;
    
    // Matrix multiplication
    if (!transA) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    float a_val = A[m * lda + k];
                    // Dequantize B: packedB is KxN row-major
                    int b_idx = k * N + n;
                    int8_t q_val = packedB[b_idx];
                    float b_val = static_cast<float>(q_val) * scaleB[n] + zeroB[n];
                    sum += a_val * b_val;
                }
                
                // Add to output with alpha scaling
                C[m * ldc + n] += alpha * sum;
            }
        }
    } else {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    float a_val = A[k * lda + m];
                    // Dequantize B: packedB is KxN row-major
                    int b_idx = k * N + n;
                    int8_t q_val = packedB[b_idx];
                    float b_val = static_cast<float>(q_val) * scaleB[n] + zeroB[n];
                    sum += a_val * b_val;
                }
                
                // Add to output with alpha scaling
                C[m * ldc + n] += alpha * sum;
            }
        }
    }
}

// Compute SGEMM with SiLU activation
void xdnn_sgemm_f32s8f32_compute_silu(bool transA, int M, int N, int K,
                                     float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                     float beta, float *C, int ldc) {
    DEBUG_PRINT();
    xdnn_sgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] = silu(C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with GELU activation
void xdnn_sgemm_f32s8f32_compute_gelu(bool transA, int M, int N, int K,
                                     float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                     float beta, float *C, int ldc) {
    DEBUG_PRINT();
    xdnn_sgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] = gelu(C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with bias addition
void xdnn_sgemm_f32s8f32_compute_biasadd(bool transA, int M, int N, int K,
                                        float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                        float beta, float *C, int ldc, const float *bias) {
    DEBUG_PRINT();
    xdnn_sgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] += bias[j];
        }
    }
}

// Compute SGEMM with bias addition and ReLU activation
void xdnn_sgemm_f32s8f32_compute_biasadd_relu(bool transA, int M, int N, int K,
                                             float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                             float beta, float *C, int ldc, const float *bias) {
    DEBUG_PRINT();
    xdnn_sgemm_f32s8f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] = std::max(0.0f, C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with residential addition
void xdnn_sgemm_f32s8f32_compute_residential(bool transA, int M, int N, int K,
                                            float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                            float beta, float *C, int ldc, const float *bias, const float *res, int ldres) {
    DEBUG_PRINT();
    xdnn_sgemm_f32s8f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] += res[i * ldres + j];
        }
    }
}

// Compute SGEMM with extended residential addition
void xdnn_sgemm_f32s8f32_compute_resext(bool transA, int M, int N, int K,
                                       float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                       float beta, float *C, int ldc, const float *bias, 
                                       float gamma, const float *res, int ldres) {
    DEBUG_PRINT();
    xdnn_sgemm_f32s8f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] += gamma * res[i * ldres + j];
        }
    }
}

// Compute SGEMM with residential multiplication
void xdnn_sgemm_f32s8f32_compute_resmul(bool transA, int M, int N, int K,
                                       float alpha, const float *A, int lda, const int8_t *packedB, const float *scaleB, const float *zeroB,
                                       float beta, float *C, int ldc, const float *res, int ldres) {
    DEBUG_PRINT();
    xdnn_sgemm_f32s8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] *= res[i * ldres + j];
        }
    }
}

// Small SGEMM for int8 input
void small_sgemm_f32s8f32(int M, int N, int K, const float *A, int lda,
                          const int8_t *B, int ldb, const float *scaleB, const float *zeroB, float *C, int ldc) {
    DEBUG_PRINT();
    // Simple implementation for small matrices
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float b_val = static_cast<float>(B[k * ldb + n]) * scaleB[n] + zeroB[n];
                sum += A[m * lda + k] * b_val;
            }
            C[m * ldc + n] = sum;
        }
    }
}