#include "conversion.h"
#include "bgemm_f32bf16f32.h"
#include "intrinsic_ext.h"
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <cmath>

// Constants for block sizes to optimize cache usage
#define BGEMM_MC 64
#define BGEMM_NC 240
#define BGEMM_KC 256

// Helper function to compute SILU activation: x * sigmoid(x)
inline float silu_activate(float x) {
    return x / (1.0f + expf(-x));
}

// Helper function to compute GELU activation
inline float gelu_activate(float x) {
    // Approximation of GELU
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Main BGEMM function that handles different matrix layouts
void xdnn_bgemm_f32bf16f32(bool transA, bool transB, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *B, int ldb,
        float beta, float *C, int ldc) {
    
    // Allocate temporary packed B matrix
    int block_rows = 16;  // Optimized for AVX-512 registers
    int block_cols = 16;
    int packedB_size = xdnn_bgemm_f32bf16f32_packb_size(N, K, block_rows, block_cols);
    XDNN_BF16 *packedB = new XDNN_BF16[packedB_size / sizeof(XDNN_BF16)];
    
    // Pack matrix B for better cache locality
    xdnn_bgemm_f32bf16f32_packb(transB, N, K, B, ldb, packedB, block_rows, block_cols);
    
    // Compute the matrix multiplication
    xdnn_bgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Free temporary packed matrix
    delete[] packedB;
}

int xdnn_bgemm_f32bf16f32_packb_size(int N, int K, int block_rows, int block_cols) {
    // Calculate the total size needed for packing matrix B
    // We need space for full blocks plus alignment padding
    int nblocks = (N + block_cols - 1) / block_cols;
    int kblocks = (K + block_rows - 1) / block_rows;
    return nblocks * kblocks * block_rows * block_cols * sizeof(XDNN_BF16);
}

void xdnn_bgemm_f32bf16f32_packb(bool transB, int N, int K, const XDNN_BF16 *B, int ldb, XDNN_BF16 *packedB, 
                                 int block_rows, int block_cols) {
    // Pack matrix B into block format for better cache usage during computation
    if (transB) {
        // B is in N x K format
        for (int j = 0; j < N; j += block_cols) {
            int jb = std::min(block_cols, N - j);
            
            for (int i = 0; i < K; i += block_rows) {
                int ib = std::min(block_rows, K - i);
                
                // Pack a single block
                for (int jj = 0; jj < jb; jj++) {
                    for (int ii = 0; ii < ib; ii++) {
                        packedB[(j/block_cols * K/block_rows + i/block_rows) * block_cols * block_rows + jj * block_rows + ii] = 
                            B[(j + jj) * ldb + (i + ii)];
                    }
                    // Zero pad if needed
                    for (int ii = ib; ii < block_rows; ii++) {
                        packedB[(j/block_cols * K/block_rows + i/block_rows) * block_cols * block_rows + jj * block_rows + ii] = 0;
                    }
                }
                
                // Zero pad if needed
                for (int jj = jb; jj < block_cols; jj++) {
                    for (int ii = 0; ii < block_rows; ii++) {
                        packedB[(j/block_cols * K/block_rows + i/block_rows) * block_cols * block_rows + jj * block_rows + ii] = 0;
                    }
                }
            }
        }
    } else {
        // B is in K x N format
        for (int i = 0; i < K; i += block_rows) {
            int ib = std::min(block_rows, K - i);
            
            for (int j = 0; j < N; j += block_cols) {
                int jb = std::min(block_cols, N - j);
                
                // Pack a single block
                for (int ii = 0; ii < ib; ii++) {
                    for (int jj = 0; jj < jb; jj++) {
                        packedB[(i/block_rows * N/block_cols + j/block_cols) * block_rows * block_cols + ii * block_cols + jj] = 
                            B[(i + ii) * ldb + (j + jj)];
                    }
                    // Zero pad if needed
                    for (int jj = jb; jj < block_cols; jj++) {
                        packedB[(i/block_rows * N/block_cols + j/block_cols) * block_rows * block_cols + ii * block_cols + jj] = 0;
                    }
                }
                
                // Zero pad if needed
                for (int ii = ib; ii < block_rows; ii++) {
                    for (int jj = 0; jj < block_cols; jj++) {
                        packedB[(i/block_rows * N/block_cols + j/block_cols) * block_rows * block_cols + ii * block_cols + jj] = 0;
                    }
                }
            }
        }
    }
}

void xdnn_bgemm_f32bf16f32_compute(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc) {
    
    // Block sizes for tiled matrix multiplication
    const int MB = BGEMM_MC;
    const int NB = BGEMM_NC;
    const int KB = BGEMM_KC;
    
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
    
    // Main computation loop with blocking for cache efficiency
    for (int i = 0; i < M; i += MB) {
        int ib = std::min(MB, M - i);
        
        for (int j = 0; j < N; j += NB) {
            int jb = std::min(NB, N - j);
            
            for (int k = 0; k < K; k += KB) {
                int kb = std::min(KB, K - k);
                
                // Compute block matrix multiplication
                for (int ii = 0; ii < ib; ii++) {
                    for (int jj = 0; jj < jb; jj++) {
                        float sum = 0.0f;
                        
                        if (transA) {
                            // A is transposed, column-major access
                            for (int kk = 0; kk < kb; kk++) {
                                sum += A[(k + kk) * lda + (i + ii)] * 
                                       _xdnn_to_float(packedB[(k/KB * N/NB + j/NB) * KB * NB + kk * NB + jj]);
                            }
                        } else {
                            // A is not transposed, row-major access
                            for (int kk = 0; kk < kb; kk++) {
                                sum += A[(i + ii) * lda + (k + kk)] * 
                                       _xdnn_to_float(packedB[(k/KB * N/NB + j/NB) * KB * NB + kk * NB + jj]);
                            }
                        }
                        
                        // Update C with accumulator and alpha scaling
                        C[(i + ii) * ldc + (j + jj)] += alpha * sum;
                    }
                }
            }
        }
    }
}

// BGEMM with SILU activation
void xdnn_bgemm_f32bf16f32_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc) {
    
    // First compute standard BGEMM
    xdnn_bgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then apply SILU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = silu_activate(C[i * ldc + j]);
        }
    }
}

// BGEMM with GELU activation
void xdnn_bgemm_f32bf16f32_compute_gelu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc) {
    
    // First compute standard BGEMM
    xdnn_bgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then apply GELU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = gelu_activate(C[i * ldc + j]);
        }
    }
}

// BGEMM with extended residual connection
void xdnn_bgemm_f32bf16f32_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc, const float *bias,
        float gamma, const float *res, int ldres) {
    
    // First compute standard BGEMM
    xdnn_bgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
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

// BGEMM with residual multiplication (element-wise)
void xdnn_bgemm_f32bf16f32_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc, const float *res, int ldres) {
    
    // First compute standard BGEMM
    xdnn_bgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then multiply by residual (element-wise)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] *= res[i * ldres + j];
        }
    }
}

// BGEMM with bias addition
void xdnn_bgemm_f32bf16f32_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc, const float *bias) {
    
    // First compute standard BGEMM
    xdnn_bgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then add bias
    if (bias) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] += bias[j];
            }
        }
    }
}

// BGEMM with bias addition and ReLU activation
void xdnn_bgemm_f32bf16f32_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc, const float *bias) {
    
    // First compute BGEMM with bias addition
    xdnn_bgemm_f32bf16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
    
    // Then apply ReLU activation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = std::max(0.0f, C[i * ldc + j]);
        }
    }
}

// BGEMM with bias addition and residual connection
void xdnn_bgemm_f32bf16f32_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_BF16 *packedB,
        float beta, float *C, int ldc, const float *bias, const float *res, int ldres) {
    
    // First compute standard BGEMM
    xdnn_bgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
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

// Small optimized BGEMM implementation for tiny matrices
void small_bgemm_f32bf16f32(int M, int N, int K, const float *A, int lda, const XDNN_BF16 *B, int ldb, float *C, int ldc) {
    // First zero initialize output
    for (int i = 0; i < M; i++) {
        memset(&C[i * ldc], 0, N * sizeof(float));
    }
    
    // Basic implementation for small matrices without blocking
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * _xdnn_to_float(B[k * ldb + j]);
            }
            
            C[i * ldc + j] = sum;
        }
    }
}
