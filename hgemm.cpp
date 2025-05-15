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

// Main HGEMM implementation with FP16 inputs and outputs
void xdnn_hgemm(bool transA, bool transB, int M, int N, int K,
                float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *B, int ldb,
                float beta, XDNN_FP16 *C, int ldc) {
    // In a real implementation, we would use optimized FP16 kernels
    // or specialized hardware instructions. For this simple version,
    // we'll convert to FP32 for computation, then back to FP16 for storage.
    
    // Apply beta scaling to C
    if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float c_val = static_cast<float>(C[i * ldc + j]) * beta;
                C[i * ldc + j] = c_val;
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
                C[i * ldc + j] = static_cast<float>(C[i * ldc + j]) + alpha * sum;
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
                C[i * ldc + j] = static_cast<float>(C[i * ldc + j]) + alpha * sum;
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
                C[i * ldc + j] = static_cast<float>(C[i * ldc + j]) + alpha * sum;
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
                C[i * ldc + j] = static_cast<float>(C[i * ldc + j]) + alpha * sum;
            }
        }
    }
}

// Pack matrix B for optimized computation
void xdnn_hgemm_packb(bool transB, int N, int K, const XDNN_FP16 *B, int ldb, XDNN_FP16 *packedB) {
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
void xdnn_hgemm_compute(bool transA, int M, int N, int K,
                       float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                       float beta, XDNN_FP16 *C, int ldc) {
    // Apply beta scaling to C
    if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float c_val = static_cast<float>(C[i * ldc + j]) * beta;
                C[i * ldc + j] = c_val;
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
                C[i * ldc + j] = static_cast<float>(C[i * ldc + j]) + alpha * sum;
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
                C[i * ldc + j] = static_cast<float>(C[i * ldc + j]) + alpha * sum;
            }
        }
    }
}

// Compute HGEMM with SiLU activation
void xdnn_hgemm_compute_silu(bool transA, int M, int N, int K,
                            float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                            float beta, XDNN_FP16 *C, int ldc) {
    // Temporary buffer for FP32 computation
    float *temp_C = new float[M * N];
    
    // Convert C to float for computation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp_C[i * N + j] = static_cast<float>(C[i * ldc + j]) * beta;
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
                temp_C[i * N + j] += alpha * sum;
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
                temp_C[i * N + j] += alpha * sum;
            }
        }
    }
    
    // Apply SiLU and convert back to FP16
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = silu(temp_C[i * N + j]);
        }
    }
    
    delete[] temp_C;
}

// Compute HGEMM with GELU activation
void xdnn_hgemm_compute_gelu(bool transA, int M, int N, int K,
                            float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                            float beta, XDNN_FP16 *C, int ldc) {
    // Temporary buffer for FP32 computation
    float *temp_C = new float[M * N];
    
    // Convert C to float for computation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp_C[i * N + j] = static_cast<float>(C[i * ldc + j]) * beta;
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
                temp_C[i * N + j] += alpha * sum;
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
                temp_C[i * N + j] += alpha * sum;
            }
        }
    }
    
    // Apply GELU and convert back to FP16
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = gelu(temp_C[i * N + j]);
        }
    }
    
    delete[] temp_C;
}

// Extended residential function
void xdnn_hgemm_compute_resext(bool transA, int M, int N, int K,
                              float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                              float beta, XDNN_FP16 *C, int ldc, const float *bias,
                              float gamma, const XDNN_FP16 *res, int ldres) {
    // Temporary buffer for FP32 computation
    float *temp_C = new float[M * N];
    
    // Convert C to float for computation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp_C[i * N + j] = static_cast<float>(C[i * ldc + j]) * beta;
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
                temp_C[i * N + j] += alpha * sum;
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
                temp_C[i * N + j] += alpha * sum;
            }
        }
    }
    
    // Add bias and residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp_C[i * N + j] += bias[j] + gamma * static_cast<float>(res[i * ldres + j]);
            C[i * ldc + j] = temp_C[i * N + j];
        }
    }
    
    delete[] temp_C;
}

// Multiplicative residential function
void xdnn_hgemm_compute_resmul(bool transA, int M, int N, int K,
                              float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                              float beta, XDNN_FP16 *C, int ldc, const XDNN_FP16 *res, int ldres) {
    // Temporary buffer for FP32 computation
    float *temp_C = new float[M * N];
    
    // Convert C to float for computation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp_C[i * N + j] = static_cast<float>(C[i * ldc + j]) * beta;
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
                temp_C[i * N + j] += alpha * sum;
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
                temp_C[i * N + j] += alpha * sum;
            }
        }
    }
    
    // Multiply by residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp_C[i * N + j] *= static_cast<float>(res[i * ldres + j]);
            C[i * ldc + j] = temp_C[i * N + j];
        }
    }
    
    delete[] temp_C;
}

// Compute HGEMM with bias addition
void xdnn_hgemm_compute_biasadd(bool transA, int M, int N, int K,
                               float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                               float beta, XDNN_FP16 *C, int ldc, const float *bias) {
    // Temporary buffer for FP32 computation
    float *temp_C = new float[M * N];
    
    // Convert C to float for computation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp_C[i * N + j] = static_cast<float>(C[i * ldc + j]) * beta;
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
                temp_C[i * N + j] += alpha * sum;
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
                temp_C[i * N + j] += alpha * sum;
            }
        }
    }
    
    // Add bias
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp_C[i * N + j] += bias[j];
            C[i * ldc + j] = temp_C[i * N + j];
        }
    }
    
    delete[] temp_C;
}

// Compute HGEMM with bias addition and ReLU activation
void xdnn_hgemm_compute_biasadd_relu(bool transA, int M, int N, int K,
                                    float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                    float beta, XDNN_FP16 *C, int ldc, const float *bias) {
    // Temporary buffer for FP32 computation
    float *temp_C = new float[M * N];
    
    // Convert C to float for computation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp_C[i * N + j] = static_cast<float>(C[i * ldc + j]) * beta;
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
                temp_C[i * N + j] += alpha * sum;
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
                temp_C[i * N + j] += alpha * sum;
            }
        }
    }
    
    // Add bias and apply ReLU
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp_C[i * N + j] += bias[j];
            // Apply ReLU
            temp_C[i * N + j] = std::max(0.0f, temp_C[i * N + j]);
            C[i * ldc + j] = temp_C[i * N + j];
        }
    }
    
    delete[] temp_C;
}

// Compute HGEMM with residential connection
void xdnn_hgemm_compute_residential(bool transA, int M, int N, int K,
                                   float alpha, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                   float beta, XDNN_FP16 *C, int ldc, const float *bias, const XDNN_FP16 *res, int ldres) {
    // Temporary buffer for FP32 computation
    float *temp_C = new float[M * N];
    
    // Convert C to float for computation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp_C[i * N + j] = static_cast<float>(C[i * ldc + j]) * beta;
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
                temp_C[i * N + j] += alpha * sum;
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
                temp_C[i * N + j] += alpha * sum;
            }
        }
    }
    
    // Add bias and residential connection
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp_C[i * N + j] += bias[j] + static_cast<float>(res[i * ldres + j]);
            C[i * ldc + j] = temp_C[i * N + j];
        }
    }
    
    delete[] temp_C;
}

// Small HGEMM implementation for single-threaded special cases
void small_hgemm(int M, int N, int K, const XDNN_FP16 *A, int lda, const XDNN_FP16 *B, int ldb, XDNN_FP16 *C, int ldc) {
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
