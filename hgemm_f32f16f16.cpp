#include "conversion.h"
#include "hgemm_f32f16f16.h"
#include "intrinsic_ext.h"
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <cmath>

// Constants for block sizes to optimize cache usage
#define HGEMM_MC 64
#define HGEMM_NC 240
#define HGEMM_KC 256

// HGEMM constants for FP16 operations
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

// Main HGEMM function that handles different matrix layouts
void xdnn_hgemm_f32f16f16(bool transA, bool transB, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *B, int ldb,
        float beta, XDNN_FP16 *C, int ldc) {
    
    // Allocate temporary packed B matrix
    size_t packedB_size = (size_t)((N + HGEMM_NR - 1) / HGEMM_NR) * 
                         ((K + HGEMM_KC - 1) / HGEMM_KC) * HGEMM_KC * HGEMM_NR * sizeof(XDNN_FP16);
    XDNN_FP16 *packedB = new XDNN_FP16[packedB_size / sizeof(XDNN_FP16)];
    
    // Pack matrix B for better cache locality
    xdnn_hgemm_f32f16f16_packb(transB, N, K, B, ldb, packedB);
    
    // Compute the matrix multiplication
    xdnn_hgemm_f32f16f16_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Free temporary packed matrix
    delete[] packedB;
}

// Pack matrix B for efficient computation
void xdnn_hgemm_f32f16f16_packb(bool transB, int N, int K, const XDNN_FP16 *B, int ldb, XDNN_FP16 *packedB) {
    // Pack matrix B into block format for better cache usage during computation
    int nb = (N + HGEMM_NR - 1) / HGEMM_NR;
    int kb = (K + HGEMM_KC - 1) / HGEMM_KC;
    
    // Zero initialize the entire packed buffer
    memset(packedB, 0, nb * kb * HGEMM_KC * HGEMM_NR * sizeof(XDNN_FP16));
    
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
                        packedB[offset + k * HGEMM_NR + n] = B[(n_block + n) * ldb + (k_block + k)];
                    }
                }
            } else {
                // B is in K x N format
                for (int k = 0; k < k_size; k++) {
                    for (int n = 0; n < n_size; n++) {
                        packedB[offset + k * HGEMM_NR + n] = B[(k_block + k) * ldb + (n_block + n)];
                    }
                }
            }
        }
    }
}

// Basic HGEMM computation with packed B matrix
void xdnn_hgemm_f32f16f16_compute(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc) {
    
    // Scale C matrix by beta if needed
    if (beta == 0.0f) {
        for (int i = 0; i < M; i++) {
            memset(&C[i * ldc], 0, N * sizeof(XDNN_FP16));
        }
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] = _xdnn_to_fp16(beta * _xdnn_to_float(C[i * ldc + j]));
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
                                sum += A[(kb + k) * lda + (m + i)] * 
                                      _xdnn_to_float(packedB[offset + k * HGEMM_NR + j]);
                            }
                        } else {
                            for (int k = 0; k < k_size; k++) {
                                sum += A[(m + i) * lda + (kb + k)] * 
                                      _xdnn_to_float(packedB[offset + k * HGEMM_NR + j]);
                            }
                        }
                        
                        // Update C with accumulated result
                        float c_val = _xdnn_to_float(C[(m + i) * ldc + (n + j)]);
                        C[(m + i) * ldc + (n + j)] = _xdnn_to_fp16(c_val + alpha * sum);
                    }
                }
            }
        }
    }
}

// HGEMM with SILU activation
void xdnn_hgemm_f32f16f16_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32f16f16_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then apply SILU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = _xdnn_to_float(C[i * ldc + j]);
            C[i * ldc + j] = _xdnn_to_fp16(silu_activate(val));
        }
    }
}

// HGEMM with GELU activation
void xdnn_hgemm_f32f16f16_compute_gelu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32f16f16_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then apply GELU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = _xdnn_to_float(C[i * ldc + j]);
            C[i * ldc + j] = _xdnn_to_fp16(gelu_activate(val));
        }
    }
}

// HGEMM with extended residual connection
void xdnn_hgemm_f32f16f16_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc, const XDNN_FP16 *bias,
        float gamma, const XDNN_FP16 *res, int ldres) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32f16f16_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then add bias and scaled residual
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = _xdnn_to_float(C[i * ldc + j]);
            
            // Add bias if provided
            if (bias) val += _xdnn_to_float(bias[j]);
            
            // Add scaled residual
            val += gamma * _xdnn_to_float(res[i * ldres + j]);
            
            C[i * ldc + j] = _xdnn_to_fp16(val);
        }
    }
}

// HGEMM with residual multiplication (element-wise)
void xdnn_hgemm_f32f16f16_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc, const XDNN_FP16 *res, int ldres) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32f16f16_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then multiply by residual (element-wise)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = _xdnn_to_float(C[i * ldc + j]);
            val *= _xdnn_to_float(res[i * ldres + j]);
            C[i * ldc + j] = _xdnn_to_fp16(val);
        }
    }
}

// HGEMM with bias addition
void xdnn_hgemm_f32f16f16_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc, const XDNN_FP16 *bias) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32f16f16_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then add bias
    if (bias) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float val = _xdnn_to_float(C[i * ldc + j]);
                val += _xdnn_to_float(bias[j]);
                C[i * ldc + j] = _xdnn_to_fp16(val);
            }
        }
    }
}

// HGEMM with bias addition and ReLU activation
void xdnn_hgemm_f32f16f16_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc, const XDNN_FP16 *bias) {
    
    // First compute HGEMM with bias addition
    xdnn_hgemm_f32f16f16_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
    
    // Then apply ReLU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = _xdnn_to_float(C[i * ldc + j]);
            val = std::max(0.0f, val);
            C[i * ldc + j] = _xdnn_to_fp16(val);
        }
    }
}

// HGEMM with bias addition and residual connection
void xdnn_hgemm_f32f16f16_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_FP16 *packedB,
        float beta, XDNN_FP16 *C, int ldc, const XDNN_FP16 *bias, const XDNN_FP16 *res, int ldres) {
    
    // First compute standard HGEMM
    xdnn_hgemm_f32f16f16_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then add bias and residual
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = _xdnn_to_float(C[i * ldc + j]);
            
            // Add bias if provided
            if (bias) val += _xdnn_to_float(bias[j]);
            
            // Add residual
            val += _xdnn_to_float(res[i * ldres + j]);
            
            C[i * ldc + j] = _xdnn_to_fp16(val);
        }
    }
}

// Small optimized HGEMM implementation for tiny matrices
void small_hgemm_f32f16f16(int M, int N, int K, const float *A, int lda, const XDNN_FP16 *B, int ldb, XDNN_FP16 *C, int ldc) {
    // First zero initialize output
    for (int i = 0; i < M; i++) {
        memset(&C[i * ldc], 0, N * sizeof(XDNN_FP16));
    }
    
    // Basic implementation for small matrices without blocking
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * _xdnn_to_float(B[k * ldb + j]);
            }
            
            C[i * ldc + j] = _xdnn_to_fp16(sum);
        }
    }
}
