#include "sgemm_f32f16f32.h"
#include "debug_print.h"
#include <vector>
#include <cmath>
#include <thread>
#include <algorithm>

// Number of threads to use for parallel operations
constexpr int NUM_THREADS = 4;

// Block sizes for matrix multiplication - these can be tuned
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;

// Helper: SiLU activation function
inline float silu(float x) {
    DEBUG_PRINT();
    return x / (1.0f + std::exp(-x));
}

// Helper: GELU activation function
inline float gelu(float x) {
    DEBUG_PRINT();
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3.0f))));
}

// Helper: ReLU activation function
inline float relu(float x) {
    DEBUG_PRINT();
    return std::max(0.0f, x);
}

// To pack matrix B (row-major KxN output)
void xdnn_sgemm_f32f16f32_packb(bool transB, int N, int K, const XDNN_FP16* B, int ldb, XDNN_FP16* packedB) {
    DEBUG_PRINT();
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            int src_idx = transB ? (n * ldb + k) : (k * ldb + n);
            int dst_idx = k * N + n;
            packedB[dst_idx] = B[src_idx];
        }
    }
}

// Basic compute function for matrix multiplication
void xdnn_sgemm_f32f16f32_compute(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_FP16* packedB,
        float beta, float* C, int ldc) {
    DEBUG_PRINT();
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                float b_val = static_cast<float>(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            // Apply alpha and beta
            if (beta == 0.0f)
                C[m * ldc + n] = alpha * sum;
            else
                C[m * ldc + n] = alpha * sum + beta * C[m * ldc + n];
        }
    }
}

// Compute with SiLU activation
void xdnn_sgemm_f32f16f32_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_FP16* packedB,
        float beta, float* C, int ldc) {
    DEBUG_PRINT();
    // First compute the matrix multiplication
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                float b_val = static_cast<float>(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            // Apply alpha and beta
            float result;
            if (beta == 0.0f)
                result = alpha * sum;
            else
                result = alpha * sum + beta * C[m * ldc + n];
                
            // Apply SiLU activation
            C[m * ldc + n] = silu(result);
        }
    }
}

// Compute with GELU activation
void xdnn_sgemm_f32f16f32_compute_gelu(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_FP16* packedB,
        float beta, float* C, int ldc) {
    DEBUG_PRINT();
    // First compute the matrix multiplication
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                float b_val = static_cast<float>(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            // Apply alpha and beta
            float result;
            if (beta == 0.0f)
                result = alpha * sum;
            else
                result = alpha * sum + beta * C[m * ldc + n];
                
            // Apply GELU activation
            C[m * ldc + n] = gelu(result);
        }
    }
}

// Compute with bias addition
void xdnn_sgemm_f32f16f32_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_FP16* packedB,
        float beta, float* C, int ldc, const float* bias) {
    DEBUG_PRINT();
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                float b_val = static_cast<float>(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            // Apply alpha, beta and add bias
            if (beta == 0.0f)
                C[m * ldc + n] = alpha * sum + bias[n];
            else
                C[m * ldc + n] = alpha * sum + beta * C[m * ldc + n] + bias[n];
        }
    }
}

// Compute with bias addition and ReLU activation
void xdnn_sgemm_f32f16f32_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_FP16* packedB,
        float beta, float* C, int ldc, const float* bias) {
    DEBUG_PRINT();
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                float b_val = static_cast<float>(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            // Apply alpha, beta, add bias, and apply ReLU
            float result;
            if (beta == 0.0f)
                result = alpha * sum + bias[n];
            else
                result = alpha * sum + beta * C[m * ldc + n] + bias[n];
                
            C[m * ldc + n] = relu(result);
        }
    }
}

// Compute with residential connections (bias + residual)
void xdnn_sgemm_f32f16f32_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_FP16* packedB,
        float beta, float* C, int ldc, const float* bias, const float* res, int ldres) {
    DEBUG_PRINT();
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                float b_val = static_cast<float>(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            // Apply alpha, beta, add bias and residual connection
            if (beta == 0.0f)
                C[m * ldc + n] = alpha * sum + bias[n] + res[m * ldres + n];
            else
                C[m * ldc + n] = alpha * sum + beta * C[m * ldc + n] + bias[n] + res[m * ldres + n];
        }
    }
}

// Extended residential computation (bias + gamma * residual)
void xdnn_sgemm_f32f16f32_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_FP16* packedB,
        float beta, float* C, int ldc, const float* bias, 
        float gamma, const float* res, int ldres) {
    DEBUG_PRINT();
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                float b_val = static_cast<float>(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            // Apply alpha, beta, add bias and scaled residual
            if (beta == 0.0f)
                C[m * ldc + n] = alpha * sum + bias[n] + gamma * res[m * ldres + n];
            else
                C[m * ldc + n] = alpha * sum + beta * C[m * ldc + n] + bias[n] + gamma * res[m * ldres + n];
        }
    }
}

// Compute with residual multiplication
void xdnn_sgemm_f32f16f32_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_FP16* packedB,
        float beta, float* C, int ldc, const float* res, int ldres) {
    DEBUG_PRINT();
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                float b_val = static_cast<float>(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            // Apply alpha and beta
            float result;
            if (beta == 0.0f)
                result = alpha * sum;
            else
                result = alpha * sum + beta * C[m * ldc + n];
                
            // Multiply by residual
            C[m * ldc + n] = result * res[m * ldres + n];
        }
    }
}

// Single-thread small SGEMM
void small_sgemm_f32f16f32(int M, int N, int K, const float* A, int lda, const XDNN_FP16* B, int ldb, float* C, int ldc) {
    DEBUG_PRINT();
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = A[m * lda + k];
                float b_val = static_cast<float>(B[k * ldb + n]);
                sum += a_val * b_val;
            }
            C[m * ldc + n] = sum;
        }
    }
}

// Worker function for parallel processing
static void compute_block(bool transA, int m_start, int m_end, int N, int K,
                  float alpha, const float* A, int lda, const XDNN_FP16* packedB,
                  float beta, float* C, int ldc) {
    DEBUG_PRINT();
    for (int m = m_start; m < m_end; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                float b_val = static_cast<float>(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            // Apply alpha and beta
            if (beta == 0.0f)
                C[m * ldc + n] = alpha * sum;
            else
                C[m * ldc + n] = alpha * sum + beta * C[m * ldc + n];
        }
    }
}

// Multi-threaded implementation of xdnn_sgemm_f32f16f32_compute
void xdnn_sgemm_f32f16f32(bool transA, bool transB, int M, int N, int K,
       float alpha, const float* A, int lda, const XDNN_FP16* B, int ldb,
       float beta, float* C, int ldc) {
    DEBUG_PRINT();
    // Create temporary storage for packed B matrix
    std::vector<XDNN_FP16> packedB(K * N);
    
    // Pack B matrix
    xdnn_sgemm_f32f16f32_packb(transB, N, K, B, ldb, packedB.data());
    
    // Use multiple threads for computation
    std::vector<std::thread> threads;
    int rows_per_thread = (M + NUM_THREADS - 1) / NUM_THREADS;
    
    for (int t = 0; t < NUM_THREADS; ++t) {
        int m_start = t * rows_per_thread;
        int m_end = std::min(M, (t + 1) * rows_per_thread);
        
        if (m_start < m_end) {
            threads.emplace_back(compute_block, transA, m_start, m_end, N, K,
                                alpha, A, lda, packedB.data(), beta, C, ldc);
        }
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

// Single-threaded version of the SGEMM function
void xdnn_sgemm_f32f16f32_single_thread(bool transA, bool transB, int M, int N, int K,
       float alpha, const float* A, int lda, const XDNN_FP16* B, int ldb,
       float beta, float* C, int ldc) {
    DEBUG_PRINT();
    // Pack B matrix
    std::vector<XDNN_FP16> packedB(K * N);
    xdnn_sgemm_f32f16f32_packb(transB, N, K, B, ldb, packedB.data());
    
    // Compute result
    xdnn_sgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB.data(), beta, C, ldc);
}