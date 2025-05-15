#include "conversion.h"
#include "sgemm_f32f16f32.h"
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

// Main SGEMM implementation with FP16 inputs
void xdnn_sgemm_f32f16f32(bool transA, bool transB, int M, int N, int K,
                          float alpha, const float *A, int lda, const XDNN_FP16 *B, int ldb,
                          float beta, float *C, int ldc) {
    // Multi-threaded implementation would typically use OpenMP or similar
    // For simplicity, we'll call the single-threaded version here
    xdnn_sgemm_f32f16f32_single_thread(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Single-threaded SGEMM implementation with FP16 inputs
void xdnn_sgemm_f32f16f32_single_thread(bool transA, bool transB, int M, int N, int K,
                                       float alpha, const float *A, int lda, const XDNN_FP16 *B, int ldb,
                                       float beta, float *C, int ldc) {
    // Apply beta scaling to C
    if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    // Matrix multiplication with alpha scaling
    if (!transA && !transB) {
        // A: M×K, B: K×N
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float temp = alpha * A[i * lda + k];
                for (int j = 0; j < N; j++) {
                    C[i * ldc + j] += temp * static_cast<float>(B[k * ldb + j]);
                }
            }
        }
    } else if (transA && !transB) {
        // A: K×M, B: K×N
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float temp = alpha * A[k * lda + i];
                for (int j = 0; j < N; j++) {
                    C[i * ldc + j] += temp * static_cast<float>(B[k * ldb + j]);
                }
            }
        }
    } else if (!transA && transB) {
        // A: M×K, B: N×K
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * lda + k] * static_cast<float>(B[j * ldb + k]);
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
                    sum += A[k * lda + i] * static_cast<float>(B[j * ldb + k]);
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// Pack matrix B for optimized computation
void xdnn_sgemm_f32f16f32_packb(bool transB, int N, int K, const XDNN_FP16 *B, int ldb, XDNN_FP16 *packedB) {
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

// Compute SGEMM with pre-packed FP16 B matrix
void xdnn_sgemm_f32f16f32_compute(bool transA, int M, int N, int K,
                                 float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
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
            for (int k = 0; k < K; k++) {
                float temp = alpha * A[i * lda + k];
                for (int j = 0; j < N; j++) {
                    C[i * ldc + j] += temp * static_cast<float>(packedB[k * N + j]);
                }
            }
        }
    } else {
        // A: K×M
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[k * lda + i] * static_cast<float>(packedB[k * N + j]);
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// Compute SGEMM with SiLU activation
void xdnn_sgemm_f32f16f32_compute_silu(bool transA, int M, int N, int K,
                                      float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
                                      float beta, float *C, int ldc) {
    // Compute regular SGEMM
    xdnn_sgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Apply SiLU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = silu(C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with GELU activation
void xdnn_sgemm_f32f16f32_compute_gelu(bool transA, int M, int N, int K,
                                      float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
                                      float beta, float *C, int ldc) {
    // Compute regular SGEMM
    xdnn_sgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Apply GELU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = gelu(C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with bias addition
void xdnn_sgemm_f32f16f32_compute_biasadd(bool transA, int M, int N, int K,
                                         float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
                                         float beta, float *C, int ldc, const float *bias) {
    // Compute regular SGEMM
    xdnn_sgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Add bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += bias[j];
        }
    }
}

// Compute SGEMM with bias addition and ReLU activation
void xdnn_sgemm_f32f16f32_compute_biasadd_relu(bool transA, int M, int N, int K,
                                              float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
                                              float beta, float *C, int ldc, const float *bias) {
    // Compute regular SGEMM with bias
    xdnn_sgemm_f32f16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
    
    // Apply ReLU
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = std::max(0.0f, C[i * ldc + j]);
        }
    }
}

// Compute SGEMM with residential connection
void xdnn_sgemm_f32f16f32_compute_residential(bool transA, int M, int N, int K,
                                             float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
                                             float beta, float *C, int ldc, const float *bias, const float *res, int ldres) {
    // Compute regular SGEMM with bias
    xdnn_sgemm_f32f16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
    
    // Add residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += res[i * ldres + j];
        }
    }
}
