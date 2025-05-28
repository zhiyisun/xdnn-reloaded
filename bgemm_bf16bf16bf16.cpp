#include "bgemm_bf16bf16bf16.h"
#include "data_types/data_types.h"
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
    return x / (1.0f + std::exp(-x));
}

// Helper: GELU activation function
inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3.0f))));
}

// Helper: ReLU activation function
inline float relu(float x) {
    return std::max(0.0f, x);
}

// Helper: Convert XDNN_BF16 to float
inline float bf16_to_float(XDNN_BF16 val) {
    union { uint32_t u; float f; } u = { static_cast<uint32_t>(val) << 16 };
    return u.f;
}

// Helper: Convert float to XDNN_BF16 (truncate lower 16 bits)
inline XDNN_BF16 float_to_bf16(float val) {
    union { float f; uint32_t u; } u = { val };
    return static_cast<XDNN_BF16>(u.u >> 16);
}

extern "C" {
// To pack matrix B (row-major KxN output)
void xdnn_bgemm_bf16bf16bf16_packb(bool transB, int N, int K, const XDNN_BF16* B, int ldb, XDNN_BF16* packedB, int block_rows, int block_cols) {
    int packed_offset = 0;
    for (int kb = 0; kb < K; kb += block_rows) {
        int kb_size = std::min(block_rows, K - kb);
        for (int nb = 0; nb < N; nb += block_cols) {
            int nb_size = std::min(block_cols, N - nb);
            for (int k = 0; k < kb_size; ++k) {
                for (int n = 0; n < nb_size; ++n) {
                    int src_k = kb + k;
                    int src_n = nb + n;
                    int src_idx = transB ? (src_n * ldb + src_k) : (src_k * ldb + src_n);
                    packedB[packed_offset++] = B[src_idx];
                }
                for (int n = nb_size; n < block_cols; ++n) {
                    packedB[packed_offset++] = 0;
                }
            }
            for (int k = kb_size; k < block_rows; ++k) {
                for (int n = 0; n < block_cols; ++n) {
                    packedB[packed_offset++] = 0;
                }
            }
        }
    }
}

void xdnn_bgemm_bf16bf16bf16_compute(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16* A, int lda, const XDNN_BF16* packedB,
        float beta, XDNN_BF16* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = bf16_to_float(transA ? A[k * lda + m] : A[m * lda + k]);
                float b_val = bf16_to_float(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            float c_val = (beta == 0.0f) ? (alpha * sum) : (alpha * sum + beta * bf16_to_float(C[m * ldc + n]));
            C[m * ldc + n] = float_to_bf16(c_val);
        }
    }
}

void xdnn_bgemm_bf16bf16bf16_compute_silu(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16* A, int lda, const XDNN_BF16* packedB,
        float beta, XDNN_BF16* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = bf16_to_float(transA ? A[k * lda + m] : A[m * lda + k]);
                float b_val = bf16_to_float(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            float result = (beta == 0.0f) ? (alpha * sum) : (alpha * sum + beta * bf16_to_float(C[m * ldc + n]));
            C[m * ldc + n] = float_to_bf16(silu(result));
        }
    }
}

void xdnn_bgemm_bf16bf16bf16_compute_gelu(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16* A, int lda, const XDNN_BF16* packedB,
        float beta, XDNN_BF16* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = bf16_to_float(transA ? A[k * lda + m] : A[m * lda + k]);
                float b_val = bf16_to_float(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            float result = (beta == 0.0f) ? (alpha * sum) : (alpha * sum + beta * bf16_to_float(C[m * ldc + n]));
            C[m * ldc + n] = float_to_bf16(gelu(result));
        }
    }
}

void xdnn_bgemm_bf16bf16bf16_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16* A, int lda, const XDNN_BF16* packedB,
        float beta, XDNN_BF16* C, int ldc, const XDNN_BF16* bias) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = bf16_to_float(transA ? A[k * lda + m] : A[m * lda + k]);
                float b_val = bf16_to_float(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            float bias_val = bf16_to_float(bias[n]);
            float c_val = (beta == 0.0f) ? (alpha * sum + bias_val) : (alpha * sum + beta * bf16_to_float(C[m * ldc + n]) + bias_val);
            C[m * ldc + n] = float_to_bf16(c_val);
        }
    }
}

void xdnn_bgemm_bf16bf16bf16_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16* A, int lda, const XDNN_BF16* packedB,
        float beta, XDNN_BF16* C, int ldc, const XDNN_BF16* bias) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = bf16_to_float(transA ? A[k * lda + m] : A[m * lda + k]);
                float b_val = bf16_to_float(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            float bias_val = bf16_to_float(bias[n]);
            float result = (beta == 0.0f) ? (alpha * sum + bias_val) : (alpha * sum + beta * bf16_to_float(C[m * ldc + n]) + bias_val);
            C[m * ldc + n] = float_to_bf16(relu(result));
        }
    }
}

void xdnn_bgemm_bf16bf16bf16_compute_residential(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16* A, int lda, const XDNN_BF16* packedB,
        float beta, XDNN_BF16* C, int ldc, const XDNN_BF16* bias, const XDNN_BF16* res, int ldres) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = bf16_to_float(transA ? A[k * lda + m] : A[m * lda + k]);
                float b_val = bf16_to_float(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            float bias_val = bf16_to_float(bias[n]);
            float res_val = bf16_to_float(res[m * ldres + n]);
            float c_val = (beta == 0.0f) ? (alpha * sum + bias_val + res_val) : (alpha * sum + beta * bf16_to_float(C[m * ldc + n]) + bias_val + res_val);
            C[m * ldc + n] = float_to_bf16(c_val);
        }
    }
}

void xdnn_bgemm_bf16bf16bf16_compute_resext(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16* A, int lda, const XDNN_BF16* packedB,
        float beta, XDNN_BF16* C, int ldc, const XDNN_BF16* bias, float gamma, const XDNN_BF16* res, int ldres) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = bf16_to_float(transA ? A[k * lda + m] : A[m * lda + k]);
                float b_val = bf16_to_float(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            float bias_val = bf16_to_float(bias[n]);
            float res_val = bf16_to_float(res[m * ldres + n]);
            float c_val = (beta == 0.0f) ? (alpha * sum + bias_val + gamma * res_val) : (alpha * sum + beta * bf16_to_float(C[m * ldc + n]) + bias_val + gamma * res_val);
            C[m * ldc + n] = float_to_bf16(c_val);
        }
    }
}

void xdnn_bgemm_bf16bf16bf16_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const XDNN_BF16* A, int lda, const XDNN_BF16* packedB,
        float beta, XDNN_BF16* C, int ldc, const XDNN_BF16* res, int ldres) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = bf16_to_float(transA ? A[k * lda + m] : A[m * lda + k]);
                float b_val = bf16_to_float(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            float result = (beta == 0.0f) ? (alpha * sum) : (alpha * sum + beta * bf16_to_float(C[m * ldc + n]));
            float res_val = bf16_to_float(res[m * ldres + n]);
            C[m * ldc + n] = float_to_bf16(result * res_val);
        }
    }
}

void small_bgemm_bf16bf16bf16(int M, int N, int K, const XDNN_BF16* A, int lda, const XDNN_BF16* B, int ldb, XDNN_BF16* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = bf16_to_float(A[m * lda + k]);
                float b_val = bf16_to_float(B[k * ldb + n]);
                sum += a_val * b_val;
            }
            C[m * ldc + n] = float_to_bf16(sum);
        }
    }
}
} // extern "C"

// Worker function for parallel processing
static void compute_block(bool transA, int m_start, int m_end, int N, int K,
                  float alpha, const XDNN_BF16* A, int lda, const XDNN_BF16* packedB,
                  float beta, XDNN_BF16* C, int ldc) {
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

// Multi-threaded implementation of xdnn_bgemm_bf16bf16bf16
void xdnn_bgemm_bf16bf16bf16(bool transA, bool transB, int M, int N, int K,
       float alpha, const XDNN_BF16* A, int lda, const XDNN_BF16* B, int ldb,
       float beta, XDNN_BF16* C, int ldc) {
    
    // Create temporary storage for packed B matrix
    std::vector<XDNN_BF16> packedB(K * N);
    
    // Pack B matrix with block-wise packing
    xdnn_bgemm_bf16bf16bf16_packb(transB, N, K, B, ldb, packedB.data(), BLOCK_K, BLOCK_N);
    
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
void xdnn_bgemm_bf16bf16bf16_single_thread(bool transA, bool transB, int M, int N, int K,
       float alpha, const XDNN_BF16* A, int lda, const XDNN_BF16* B, int ldb,
       float beta, XDNN_BF16* C, int ldc) {
    
    // Pack B matrix with block-wise packing
    std::vector<XDNN_BF16> packedB(K * N);
    xdnn_bgemm_bf16bf16bf16_packb(transB, N, K, B, ldb, packedB.data(), BLOCK_K, BLOCK_N);
    
    // Compute result
    xdnn_bgemm_bf16bf16bf16_compute(transA, M, N, K, alpha, A, lda, packedB.data(), beta, C, ldc);
}