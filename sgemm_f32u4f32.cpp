#include "conversion.h"
#include "sgemm_f32u4f32.h"
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

// Helper functions to get and set 4-bit values
static uint8_t get_u4_val_static(const XDNN_UINT4x2* data, int index) {
    DEBUG_PRINT();
    const uint8_t* byte_data = reinterpret_cast<const uint8_t*>(data);
    uint8_t packed_byte = byte_data[index / 2];
    if (index % 2 == 0) {
        return packed_byte & 0x0F; // Lower nibble
    } else {
        return (packed_byte >> 4) & 0x0F; // Upper nibble
    }
}

// Helper to set individual uint4 values into XDNN_UINT4x2
static void set_u4_val_static(XDNN_UINT4x2* data, int index, uint8_t val) {
    DEBUG_PRINT();
    uint8_t* byte_data = reinterpret_cast<uint8_t*>(data);
    int byte_idx = index / 2;
    uint8_t current_byte = byte_data[byte_idx];
    if (index % 2 == 0) { // Lower nibble
        byte_data[byte_idx] = (current_byte & 0xF0) | (val & 0x0F);
    } else { // Upper nibble
        byte_data[byte_idx] = (current_byte & 0x0F) | ((val & 0x0F) << 4);
    }
}

// Quantize FP32 to UINT4 (4-bit unsigned integer)
void xdnn_sgemm_f32u4f32_quantize(bool transB, int N, int K, const float *B, int ldb,
                                 float quantization_rate, XDNN_UINT4x2 *quantizedB, int ldqb, float *scaleB, float *zeroB) {

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
        
        // Apply quantization rate if specified
        if (quantization_rate < 1.0f && col_max > col_min) {
            float center = (col_max + col_min) / 2.0f;
            float half_range = (col_max - col_min) / 2.0f * quantization_rate;
            col_min = center - half_range;
            col_max = center + half_range;
        }
        
        // Calculate scale (avoid division by zero)
        float scale = (col_max == col_min) ? 1.0f : (col_max - col_min) / 16.0f; // Use 16.0f to match reference
        if (scale == 0.0f) scale = 1e-9f;
        
        scaleB[j] = scale;
        zeroB[j] = col_min;
    }
    
    // Quantize values
    for (int i = 0; i < num_quant_rows; ++i) {
        for (int j = 0; j < num_quant_cols; ++j) {
            float val = transB ? B[j * ldb + i] : B[i * ldb + j];
            float col_min = zeroB[j];
            float scale = scaleB[j];
            
            // Clip value to range [col_min, col_min + 16*scale]
            float val_clipped = std::max(col_min, std::min(val, col_min + 16.0f * scale));
            
            // Quantize value
            float q = std::round((val_clipped - col_min) / scale);
            uint8_t quantized_val = static_cast<uint8_t>(std::max(0.0f, std::min(15.0f, q)));
            
            // Store in quantizedB using row-major layout (to match reference implementation)
            int u4_index = i * num_quant_cols + j;
            set_u4_val_static(quantizedB, u4_index, quantized_val);
        }
    }
}

// Helper function to dequantize UINT4x2 to float
inline void uint4x2_to_float(const XDNN_UINT4x2& u4x2, float scale, float zero, float& val1, float& val2) {
    val1 = u4x2.get_v1() * scale + zero;
    val2 = u4x2.get_v2() * scale + zero;
}

// Main SGEMM implementation with UINT4 quantized B matrix
void xdnn_sgemm_f32u4f32(bool transA, bool transB, int M, int N, int K,
                        float alpha, const float *A, int lda, const XDNN_UINT4x2 *B, int ldb, const float *scaleB, const float *zeroB,
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
    // K is the original K, but actual packed K is (K+1)/2 (since each UINT4x2 contains two values)
    int packed_K = (K + 1) / 2;
    
    if (!transA && !transB) {
        // A: M×K, B: K×N
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int pk = 0; pk < packed_K; pk++) {
                    int k = pk * 2;
                    float b_val1, b_val2;
                    uint4x2_to_float(B[pk * ldb + j], scaleB[j], zeroB[j], b_val1, b_val2);
                    
                    sum += A[i * lda + k] * b_val1;
                    if (k + 1 < K) {
                        sum += A[i * lda + k + 1] * b_val2;
                    }
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else if (transA && !transB) {
        // A: K×M, B: K×N
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int pk = 0; pk < packed_K; pk++) {
                    int k = pk * 2;
                    float b_val1, b_val2;
                    uint4x2_to_float(B[pk * ldb + j], scaleB[j], zeroB[j], b_val1, b_val2);
                    
                    sum += A[k * lda + i] * b_val1;
                    if (k + 1 < K) {
                        sum += A[(k + 1) * lda + i] * b_val2;
                    }
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else if (!transA && transB) {
        // A: M×K, B: N×K
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int pk = 0; pk < packed_K; pk++) {
                    int k = pk * 2;
                    float b_val1, b_val2;
                    uint4x2_to_float(B[j * ldb + pk], scaleB[j], zeroB[j], b_val1, b_val2);
                    
                    sum += A[i * lda + k] * b_val1;
                    if (k + 1 < K) {
                        sum += A[i * lda + k + 1] * b_val2;
                    }
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else { // transA && transB
        // A: K×M, B: N×K
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int pk = 0; pk < packed_K; pk++) {
                    int k = pk * 2;
                    float b_val1, b_val2;
                    uint4x2_to_float(B[j * ldb + pk], scaleB[j], zeroB[j], b_val1, b_val2);
                    
                    sum += A[k * lda + i] * b_val1;
                    if (k + 1 < K) {
                        sum += A[(k + 1) * lda + i] * b_val2;
                    }
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// Pack matrix B for optimized computation
void xdnn_sgemm_f32u4f32_packb(bool transB, int N, int K, const XDNN_UINT4x2 *B, int ldb, XDNN_UINT4x2 *packedB) {
    DEBUG_PRINT();
    // Reference: see reference_packb_u4 in test_sgemm_f32u4f32.cpp
    // Output packedB is row-major KxN, tightly packed 4-bit
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            int src_idx = transB ? (n * ldb + k) : (k * ldb + n);
            int dst_idx = k * N + n;
            uint8_t val = get_u4_val_static(B, src_idx);
            set_u4_val_static(packedB, dst_idx, val);
        }
    }
}

// Compute SGEMM with pre-packed UINT4 B matrix
void xdnn_sgemm_f32u4f32_compute(bool transA, int M, int N, int K,
                                float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                float beta, float *C, int ldc) {
    DEBUG_PRINT();
    // This implementation is based on reference_sgemm_f32u4f32_compute in test_sgemm_f32u4f32.cpp
    
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
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                // Dequantize B: packedB is KxN row-major, 4-bit per value
                int b_idx = k * N + n;
                uint8_t q_val = get_u4_val_static(packedB, b_idx);
                float b_val = scaleB[n] * q_val + zeroB[n];
                sum += a_val * b_val;
            }
            
            // Add to output with alpha scaling
            C[m * ldc + n] += alpha * sum;
        }
    }
}

// Compute SGEMM with SiLU activation
void xdnn_sgemm_f32u4f32_compute_silu(bool transA, int M, int N, int K,
                                     float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                     float beta, float *C, int ldc) {
    DEBUG_PRINT();
    xdnn_sgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] = silu(C[i * ldc + j]);
        }
    }
}

void xdnn_sgemm_f32u4f32_compute_gelu(bool transA, int M, int N, int K,
                                     float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                     float beta, float *C, int ldc) {
    DEBUG_PRINT();
    xdnn_sgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] = gelu(C[i * ldc + j]);
        }
    }
}

void xdnn_sgemm_f32u4f32_compute_biasadd(bool transA, int M, int N, int K,
                                        float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                        float beta, float *C, int ldc, const float *bias) {
    DEBUG_PRINT();
    xdnn_sgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] += bias[j];
        }
    }
}

void xdnn_sgemm_f32u4f32_compute_biasadd_relu(bool transA, int M, int N, int K,
                                             float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                             float beta, float *C, int ldc, const float *bias) {
    DEBUG_PRINT();
    xdnn_sgemm_f32u4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] = std::max(0.0f, C[i * ldc + j]);
        }
    }
}

void xdnn_sgemm_f32u4f32_compute_residential(bool transA, int M, int N, int K,
                                            float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                            float beta, float *C, int ldc, const float *bias, const float *res, int ldres) {
    DEBUG_PRINT();
    xdnn_sgemm_f32u4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] += res[i * ldres + j];
        }
    }
}

void xdnn_sgemm_f32u4f32_compute_resext(bool transA, int M, int N, int K,
                                       float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                       float beta, float *C, int ldc, const float *bias, 
                                       float gamma, const float *res, int ldres) {
    DEBUG_PRINT();
    xdnn_sgemm_f32u4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] += gamma * res[i * ldres + j];
        }
    }
}

void xdnn_sgemm_f32u4f32_compute_resmul(bool transA, int M, int N, int K,
                                       float alpha, const float *A, int lda, const XDNN_UINT4x2 *packedB, const float *scaleB, const float *zeroB,
                                       float beta, float *C, int ldc, const float *res, int ldres) {
    DEBUG_PRINT();
    xdnn_sgemm_f32u4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] *= res[i * ldres + j];
        }
    }
}
