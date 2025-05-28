#include "conversion.h"
#include "hgemm.h"
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

// Main hgemm implementation with multi-threading support
void xdnn_hgemm(bool transA, bool transB, int M, int N, int K,
                float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *B, int ldb,
                float beta, XDNN_FP16 *C, int ldc) {
    // Apply beta scaling to C
    if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float cval = static_cast<float>(C[i * ldc + j]);
                C[i * ldc + j] = XDNN_FP16(cval * beta);
            }
        }
    }
    // Matrix multiplication with alpha scaling
    if (!transA && !transB) {
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float aval = static_cast<float>(A[i * lda + k]);
                float temp = alpha * aval;
                for (int j = 0; j < N; j++) {
                    float bval = static_cast<float>(B[k * ldb + j]);
                    float cval = static_cast<float>(C[i * ldc + j]);
                    C[i * ldc + j] = XDNN_FP16(cval + temp * bval);
                }
            }
        }
    } else if (transA && !transB) {
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float aval = static_cast<float>(A[k * lda + i]);
                float temp = alpha * aval;
                for (int j = 0; j < N; j++) {
                    float bval = static_cast<float>(B[k * ldb + j]);
                    float cval = static_cast<float>(C[i * ldc + j]);
                    C[i * ldc + j] = XDNN_FP16(cval + temp * bval);
                }
            }
        }
    } else if (!transA && transB) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    float aval = static_cast<float>(A[i * lda + k]);
                    float bval = static_cast<float>(B[j * ldb + k]);
                    sum += aval * bval;
                }
                float cval = static_cast<float>(C[i * ldc + j]);
                C[i * ldc + j] = XDNN_FP16(cval + alpha * sum);
            }
        }
    } else { // transA && transB
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    float aval = static_cast<float>(A[k * lda + i]);
                    float bval = static_cast<float>(B[j * ldb + k]);
                    sum += aval * bval;
                }
                float cval = static_cast<float>(C[i * ldc + j]);
                C[i * ldc + j] = XDNN_FP16(cval + alpha * sum);
            }
        }
    }
}

// Pack matrix B for optimized computation
void xdnn_hgemm_packb(bool transB, int N, int K, const XDNN_FP16 *B, int ldb, XDNN_FP16 *packedB) {
    // Packing B for better cache locality in subsequent computations
    // The exact packing format depends on the target architecture and SIMD width
    
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

// Compute hgemm with pre-packed B matrix
void xdnn_hgemm_compute(bool transA, int M, int N, int K,
                        float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                        float beta, XDNN_FP16 *C, int ldc) {
    // Apply beta scaling to C
    if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float cval = static_cast<float>(C[i * ldc + j]);
                C[i * ldc + j] = XDNN_FP16(cval * beta);
            }
        }
    }
    // Matrix multiplication with pre-packed B
    if (!transA) {
        // A: M×K
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                float aval = static_cast<float>(A[i * lda + k]);
                float temp = alpha * aval;
                for (int j = 0; j < N; j++) {
                    float bval = static_cast<float>(packedB[k * N + j]);
                    float cval = static_cast<float>(C[i * ldc + j]);
                    C[i * ldc + j] = XDNN_FP16(cval + temp * bval);
                }
            }
        }
    } else {
        // A: K×M
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    float aval = static_cast<float>(A[k * lda + i]);
                    float bval = static_cast<float>(packedB[k * N + j]);
                    sum += aval * bval;
                }
                float cval = static_cast<float>(C[i * ldc + j]);
                C[i * ldc + j] = XDNN_FP16(cval + alpha * sum);
            }
        }
    }
}

// Compute hgemm with SiLU activation
void xdnn_hgemm_compute_silu(bool transA, int M, int N, int K,
                             float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                             float beta, XDNN_FP16 *C, int ldc) {
    // Compute regular hgemm
    xdnn_hgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Apply SiLU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float v = static_cast<float>(C[i * ldc + j]);
            C[i * ldc + j] = XDNN_FP16(silu(v));
        }
    }
}

// Compute hgemm with GELU activation
void xdnn_hgemm_compute_gelu(bool transA, int M, int N, int K,
                             float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                             float beta, XDNN_FP16 *C, int ldc) {
    // Compute regular hgemm
    xdnn_hgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Apply GELU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float v = static_cast<float>(C[i * ldc + j]);
            C[i * ldc + j] = XDNN_FP16(gelu(v));
        }
    }
}

// Compute hgemm with bias addition
void xdnn_hgemm_compute_biasadd(bool transA, int M, int N, int K,
                               float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                               float beta, XDNN_FP16 *C, int ldc, const float *bias) {
    // Compute regular hgemm
    xdnn_hgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Add bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float v = static_cast<float>(C[i * ldc + j]) + bias[j];
            C[i * ldc + j] = XDNN_FP16(v);
        }
    }
}

// Compute hgemm with bias addition and ReLU activation
void xdnn_hgemm_compute_biasadd_relu(bool transA, int M, int N, int K,
                                     float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                     float beta, XDNN_FP16 *C, int ldc, const float *bias) {
    // Compute regular hgemm with bias
    xdnn_hgemm_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
    
    // Apply ReLU
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float v = static_cast<float>(C[i * ldc + j]);
            C[i * ldc + j] = XDNN_FP16(std::max(0.0f, v));
        }
    }
}

// Compute hgemm with residential connection
void xdnn_hgemm_compute_residential(bool transA, int M, int N, int K,
                                    float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                    float beta, XDNN_FP16 *C, int ldc, const float *bias, const XDNN_FP16 *res, int ldres) {
    // Compute regular hgemm with bias
    xdnn_hgemm_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
    
    // Add residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float v = static_cast<float>(C[i * ldc + j]) + static_cast<float>(res[i * ldres + j]);
            C[i * ldc + j] = XDNN_FP16(v);
        }
    }
}

// Compute hgemm with extended residential connection (assumed to be addition)
void xdnn_hgemm_compute_resext(bool transA, int M, int N, int K,
                                   float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                   float beta, XDNN_FP16 *C, int ldc, const float *bias, float gamma, const XDNN_FP16 *res, int ldres) {
    // Compute regular hgemm with bias
    xdnn_hgemm_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
    
    // Add residential connection scaled by gamma
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float v = static_cast<float>(C[i * ldc + j]) + gamma * static_cast<float>(res[i * ldres + j]);
            C[i * ldc + j] = XDNN_FP16(v);
        }
    }
}

// Compute hgemm with residential multiplication
void xdnn_hgemm_compute_resmul(bool transA, int M, int N, int K,
                                   float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                   float beta, XDNN_FP16 *C, int ldc, const XDNN_FP16 *res, int ldres) {
    // Compute regular hgemm
    xdnn_hgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Multiply by residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float v = static_cast<float>(C[i * ldc + j]) * static_cast<float>(res[i * ldres + j]);
            C[i * ldc + j] = XDNN_FP16(v);
        }
    }
}

// ================================================================================
// Below is single thread small hgemm
// ================================================================================
void small_hgemm(int M, int N, int K, const XDNN_FP16 *A, int lda, const XDNN_FP16 *B, int ldb, XDNN_FP16 *C, int ldc) {
    // Assuming A, B are not transposed (transA=false, transB=false)
    // Assuming alpha = 1.0f and beta = 0.0f (C is overwritten)

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float aval = static_cast<float>(A[i * lda + k]);
                float bval = static_cast<float>(B[k * ldb + j]);
                sum += aval * bval;
            }
            C[i * ldc + j] = XDNN_FP16(sum);
        }
    }
}
