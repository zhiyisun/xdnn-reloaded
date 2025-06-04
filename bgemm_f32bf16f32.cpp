#include "bgemm_f32bf16f32.h"
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
void xdnn_bgemm_f32bf16f32_packb(bool transB, int N, int K, const XDNN_BF16* B, int ldb, XDNN_BF16* packedB, int block_rows, int block_cols) {
    // DEBUG_PRINT();
    DEBUG_PRINT_PARAMS("transB = %d, N = %d, K = %d, ldb = %d, block_rows = %d, block_cols = %d\n", transB, N, K, ldb, block_rows, block_cols);
    std::vector<XDNN_BF16> B_buf;
    const XDNN_BF16* B_used = B;
    if (transB) {
        // Transpose B (original shape KxN, ldb)
        B_buf.resize(K * N, 0);
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < K; ++c) {
                B_buf[c * N + r] = B[r * K + c];
            }
        }
        B_used = B_buf.data();
    }
    int idx = 0;
    int packed_idx = 0;
    int packed_row_per_block = 0;
    if ((K / 2) > block_rows) {
        packed_row_per_block = K / 2;
    }
    else {
        packed_row_per_block = block_rows;
    }

    int packed_cols = block_cols * 2;
    int packed_rows_per_rowB = N / block_cols;

    for (int row = 0; row < K; ++ row) {
        for (int col = 0; col < N; ++ col) {
            idx = row * N + col;
            int pos_in_packed_row = 2 * (idx % block_cols) + (idx / N) % 2;
            int block_per_rowB = (idx % N) / block_cols;
            int packed_row_block_offset = block_per_rowB * packed_row_per_block;
            int packed_row_offset_in_block = idx / (N * 2);
            packed_idx = pos_in_packed_row + (packed_row_block_offset + packed_row_offset_in_block) * block_cols * 2;
            packedB[packed_idx] = B_used[idx];
        }
    }
}

// Basic compute function for matrix multiplication
void xdnn_bgemm_f32bf16f32_compute(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_BF16* packedB,
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
void xdnn_bgemm_f32bf16f32_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_BF16* packedB,
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
            float result;
            if (beta == 0.0f)
                result = alpha * sum;
            else
                result = alpha * sum + beta * C[m * ldc + n];
            C[m * ldc + n] = silu(result);
        }
    }
}

// Compute with GELU activation
void xdnn_bgemm_f32bf16f32_compute_gelu(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_BF16* packedB,
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
            float result;
            if (beta == 0.0f)
                result = alpha * sum;
            else
                result = alpha * sum + beta * C[m * ldc + n];
            C[m * ldc + n] = gelu(result);
        }
    }
}

// Compute with bias addition
void xdnn_bgemm_f32bf16f32_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_BF16* packedB,
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
            if (beta == 0.0f)
                C[m * ldc + n] = alpha * sum + bias[n];
            else
                C[m * ldc + n] = alpha * sum + beta * C[m * ldc + n] + bias[n];
        }
    }
}

// Compute with bias addition and ReLU activation
void xdnn_bgemm_f32bf16f32_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_BF16* packedB,
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
void xdnn_bgemm_f32bf16f32_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_BF16* packedB,
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
            if (beta == 0.0f)
                C[m * ldc + n] = alpha * sum + bias[n] + res[m * ldres + n];
            else
                C[m * ldc + n] = alpha * sum + beta * C[m * ldc + n] + bias[n] + res[m * ldres + n];
        }
    }
}

// Extended residential computation (bias + gamma * residual)
void xdnn_bgemm_f32bf16f32_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_BF16* packedB,
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
            if (beta == 0.0f)
                C[m * ldc + n] = alpha * sum + bias[n] + gamma * res[m * ldres + n];
            else
                C[m * ldc + n] = alpha * sum + beta * C[m * ldc + n] + bias[n] + gamma * res[m * ldres + n];
        }
    }
}

// Compute with residual multiplication
void xdnn_bgemm_f32bf16f32_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float* A, int lda, const XDNN_BF16* packedB,
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
            float result;
            if (beta == 0.0f)
                result = alpha * sum;
            else
                result = alpha * sum + beta * C[m * ldc + n];
            C[m * ldc + n] = result * res[m * ldres + n];
        }
    }
}

// Single-thread small BGEMM
void small_bgemm_f32bf16f32(int M, int N, int K, const float* A, int lda, const XDNN_BF16* B, int ldb, float* C, int ldc) {
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
                  float alpha, const float* A, int lda, const XDNN_BF16* packedB,
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

// Multi-threaded implementation of xdnn_bgemm_f32bf16f32_compute
void xdnn_bgemm_f32bf16f32(bool transA, bool transB, int M, int N, int K,
       float alpha, const float* A, int lda, const XDNN_BF16* B, int ldb,
       float beta, float* C, int ldc) {
    
    DEBUG_PRINT();
    // Create temporary storage for packed B matrix
    std::vector<XDNN_BF16> packedB(K * N);
    
    // Pack B matrix with block-wise packing
    xdnn_bgemm_f32bf16f32_packb(transB, N, K, B, ldb, packedB.data(), BLOCK_K, BLOCK_N);
    
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

// Single-threaded version of the BGEMM function
void xdnn_bgemm_f32bf16f32_single_thread(bool transA, bool transB, int M, int N, int K,
       float alpha, const float* A, int lda, const XDNN_BF16* B, int ldb,
       float beta, float* C, int ldc) {
    
    DEBUG_PRINT();
    // Pack B matrix with block-wise packing
    std::vector<XDNN_BF16> packedB(K * N);
    xdnn_bgemm_f32bf16f32_packb(transB, N, K, B, ldb, packedB.data(), BLOCK_K, BLOCK_N);
    
    // Compute result
    xdnn_bgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB.data(), beta, C, ldc);
}