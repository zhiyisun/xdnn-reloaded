#include "conversion.h"
#include "hgemm_f16f16f32.h"
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

// Main HGEMM implementation with FP16 inputs and FP32 output
void xdnn_hgemm_f16f16f32(bool transA, bool transB, int M, int N, int K,
                         float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *B, int ldb,
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
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += static_cast<float>(A[i * lda + k]) * static_cast<float>(B[k * ldb + j]);
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
                    sum += static_cast<float>(A[k * lda + i]) * static_cast<float>(B[k * ldb + j]);
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
                    sum += static_cast<float>(A[i * lda + k]) * static_cast<float>(B[j * ldb + k]);
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
                    sum += static_cast<float>(A[k * lda + i]) * static_cast<float>(B[j * ldb + k]);
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// Pack matrix B for optimized computation
void xdnn_hgemm_f16f16f32_packb(bool transB, int N, int K, const XDNN_FP16 *B, int ldb, XDNN_FP16 *packedB) {
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

// Compute HGEMM with pre-packed B matrix
void xdnn_hgemm_f16f16f32_compute(bool transA, int M, int N, int K,
                                 float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
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
                    sum += static_cast<float>(A[i * lda + k]) * static_cast<float>(packedB[k * N + j]);
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
                    sum += static_cast<float>(A[k * lda + i]) * static_cast<float>(packedB[k * N + j]);
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// Compute HGEMM with SiLU activation
void xdnn_hgemm_f16f16f32_compute_silu(bool transA, int M, int N, int K,
                                      float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                      float beta, float *C, int ldc) {
    // Compute regular HGEMM
    xdnn_hgemm_f16f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Apply SiLU
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = silu(C[i * ldc + j]);
        }
    }
}

// Compute HGEMM with GELU activation
void xdnn_hgemm_f16f16f32_compute_gelu(bool transA, int M, int N, int K,
                                      float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                      float beta, float *C, int ldc) {
    // Compute regular HGEMM
    xdnn_hgemm_f16f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Apply GELU
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = gelu(C[i * ldc + j]);
        }
    }
}

// Extended residential function
void xdnn_hgemm_f16f16f32_compute_resext(bool transA, int M, int N, int K,
                                        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                        float beta, float *C, int ldc, const float *bias,
                                        float gamma, const float *res, int ldres) {
    // Compute regular HGEMM
    xdnn_hgemm_f16f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Add bias and scaled residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += bias[j] + gamma * res[i * ldres + j];
        }
    }
}

// Multiplicative residential function
void xdnn_hgemm_f16f16f32_compute_resmul(bool transA, int M, int N, int K,
                                        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                        float beta, float *C, int ldc, const float *res, int ldres) {
    // Compute regular HGEMM
    xdnn_hgemm_f16f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Multiply by residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] *= res[i * ldres + j];
        }
    }
}

// Compute HGEMM with bias addition
void xdnn_hgemm_f16f16f32_compute_biasadd(bool transA, int M, int N, int K,
                                         float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                         float beta, float *C, int ldc, const float *bias) {
    // Compute regular HGEMM
    xdnn_hgemm_f16f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Add bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += bias[j];
        }
    }
}

// Compute HGEMM with bias addition and ReLU activation
void xdnn_hgemm_f16f16f32_compute_biasadd_relu(bool transA, int M, int N, int K,
                                              float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                              float beta, float *C, int ldc, const float *bias) {
    // Compute regular HGEMM with bias addition
    xdnn_hgemm_f16f16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
    
    // Apply ReLU
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = std::max(0.0f, C[i * ldc + j]);
        }
    }
}

// Compute HGEMM with residential connection
void xdnn_hgemm_f16f16f32_compute_residential(bool transA, int M, int N, int K,
                                             float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                             float beta, float *C, int ldc, const float *bias, const float *res, int ldres) {
    // Compute regular HGEMM with bias addition
    xdnn_hgemm_f16f16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
    
    // Add residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] += res[i * ldres + j];
        }
    }
}

// Small HGEMM implementation for single-threaded special cases
void small_hgemm_f16f16f32(int M, int N, int K, const XDNN_FP16 *A, int lda, const XDNN_FP16 *B, int ldb, float *C, int ldc) {
    // Simple implementation optimized for small matrices
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += static_cast<float>(A[i * lda + k]) * static_cast<float>(B[k * ldb + j]);
            }
            C[i * ldc + j] = sum;
        }
    }
}
