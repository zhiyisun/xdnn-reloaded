// NOTE: The quantize function in the original library may not be correct. Please review its logic carefully before use.
//
// This file implements SGEMM with NF4 quantization and various post-processing options.
// It includes functions for quantization, matrix packing, and matrix multiplication with support
// for different activation functions and residual connections.

#include "sgemm_f32nf4f32.h"
#include "debug_print.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

// Helper function to get individual nf4 values from XDNN_NF4x2
static uint8_t get_nf4_val(const XDNN_NF4x2* data, int index) {
    DEBUG_PRINT();
    const uint8_t* byte_data = reinterpret_cast<const uint8_t*>(data);
    uint8_t packed_byte = byte_data[index / 2];
    if (index % 2 == 0) {
        return packed_byte & 0x0F; // Extract lower 4 bits
    } else {
        return (packed_byte >> 4) & 0x0F; // Extract upper 4 bits
    }
}

// Helper to set individual nf4 values into XDNN_NF4x2
static void set_nf4_val(XDNN_NF4x2* data, int index, uint8_t val) {
    DEBUG_PRINT();
    uint8_t* byte_data = reinterpret_cast<uint8_t*>(data);
    int byte_idx = index / 2;
    uint8_t current_byte = byte_data[byte_idx];
    if (index % 2 == 0) {
        // Set lower 4 bits, preserve upper 4
        byte_data[byte_idx] = (current_byte & 0xF0) | (val & 0x0F);
    } else {
        // Set upper 4 bits, preserve lower 4
        byte_data[byte_idx] = (current_byte & 0x0F) | ((val & 0x0F) << 4);
    }
}

extern "C" {

// Symmetric Quantization per Columns
void xdnn_sgemm_f32nf4f32_quantize(bool transB, int N, int K, const float *B, int ldb,
        float quantization_rate, XDNN_NF4x2 *quantizedB, int ldqb, float *scaleB, float *zeroB) {
    DEBUG_PRINT();
    // If input dimensions are invalid, nothing to do
    if (N <= 0 || K <= 0) {
        return;
    }
    
    const int num_quant_cols = transB ? K : N;
    const int num_quant_rows = transB ? N : K;
    
    // If either dimension is zero, nothing to quantize
    if (num_quant_cols <= 0 || num_quant_rows <= 0) {
        for (int j = 0; j < num_quant_cols; ++j) {
            scaleB[j] = 0.0f;
            zeroB[j] = 0.0f;
        }
        return;
    }
    
    // Process each column to calculate scale and zero-point
    for (int j = 0; j < num_quant_cols; ++j) {
        float col_min = std::numeric_limits<float>::max();
        float col_max = std::numeric_limits<float>::lowest();
        
        // Find min and max values in the column
        for (int i = 0; i < num_quant_rows; ++i) {
            float val = transB ? B[j * ldb + i] : B[i * ldb + j];
            col_min = std::min(col_min, val);
            col_max = std::max(col_max, val);
        }
        
        // For Normal Float 4-bit (NF4), determine scale
        float abs_max = std::max(std::abs(col_min), std::abs(col_max));
        float scale = (abs_max == 0.0f) ? 1e-9f : abs_max;
        
        // Apply quantization_rate - this can be used to clip outliers
        if (quantization_rate < 1.0f && quantization_rate > 0.0f) {
            scale = scale / quantization_rate;
        }
        
        // Store scale and zero point
        scaleB[j] = scale;
        zeroB[j] = 0.0f;  // NF4 is centered around zero
    }
    
    // Quantize the values
    for (int i = 0; i < num_quant_rows; ++i) {
        for (int j = 0; j < num_quant_cols; ++j) {
            float original_val = transB ? B[j * ldb + i] : B[i * ldb + j];
            float scale = scaleB[j];
            float normalized_val = original_val / scale;
            
            // Find the closest value in XDNN_NORMAL_FLOAT32 array
            uint8_t quantized_val_u8 = 0;
            float min_diff = std::abs(std::abs(normalized_val) - std::abs(XDNN_NORMAL_FLOAT32[0]));
            
            for (int idx = 1; idx < 16; ++idx) {
                float diff = std::abs(std::abs(normalized_val) - std::abs(XDNN_NORMAL_FLOAT32[idx]));
                if (diff < min_diff) {
                    min_diff = diff;
                    quantized_val_u8 = static_cast<uint8_t>(idx);
                }
            }
            
            // Store quantized value - row-major storage
            int nf4_index = i * num_quant_cols + j;
            set_nf4_val(quantizedB, nf4_index, quantized_val_u8);
        }
    }
}

// Pack matrix B for optimized computation
void xdnn_sgemm_f32nf4f32_packb(bool transB, int N, int K, const XDNN_NF4x2 *B, int ldb, XDNN_NF4x2 *packedB) {
    DEBUG_PRINT();
    // Output is always KxN, row-major, tightly packed 4-bit
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            int src_idx = transB ? (n * ldb + k) : (k * ldb + n);
            int dst_idx = k * N + n;
            uint8_t val = get_nf4_val(B, src_idx);
            set_nf4_val(packedB, dst_idx, val);
        }
    }
}

// Basic compute function for SGEMM with NF4 quantization
void xdnn_sgemm_f32nf4f32_compute(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc) {
    DEBUG_PRINT();
    // Implementation constraints
    if (!(alpha == 1.0f)) {
        std::cerr << "WARNING: xdnn_sgemm_f32nf4f32_compute only supports alpha = 1.0f" << std::endl;
        return;
    }
    
    if (!(beta == 0.0f || beta == 1.0f)) {
        std::cerr << "WARNING: xdnn_sgemm_f32nf4f32_compute only supports beta = 0.0f or 1.0f" << std::endl;
        return;
    }
    
    // Empty matrix case
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }
    
    // Compute matrix multiplication
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // Get A value
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                
                // Dequantize B: packedB is KxN row-major
                int b_idx = k * N + n;
                uint8_t q_val = get_nf4_val(packedB, b_idx);
                float b_val = scaleB[n] * XDNN_NORMAL_FLOAT32[q_val] + zeroB[n];
                
                // Multiply and accumulate
                sum += a_val * b_val;
            }
            
            // Apply beta (either 0 or 1)
            if (beta == 0.0f) {
                C[m * ldc + n] = sum;
            } else {
                C[m * ldc + n] = sum + C[m * ldc + n];
            }
        }
    }
}

// Main SGEMM function that combines quantization, packing and computation
void xdnn_sgemm_f32nf4f32(bool transA, bool transB, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_NF4x2 *B, int ldb, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc) {
    DEBUG_PRINT();
    // This function combines quantize + packb + compute
    // For now, simply call the compute function, assuming B is already quantized and packed
    xdnn_sgemm_f32nf4f32_compute(transA, M, N, K, alpha, A, lda, B, scaleB, zeroB, beta, C, ldc);
}

// Helper function for applying SiLU activation: x / (1 + exp(-x))
static float silu_activation(float x) {
    return x / (1.0f + std::exp(-x));
}

// Compute with SiLU activation
void xdnn_sgemm_f32nf4f32_compute_silu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc) {
    DEBUG_PRINT();
    
    // First compute the regular matrix multiplication
    xdnn_sgemm_f32nf4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Then apply SiLU activation
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * ldc + n] = silu_activation(C[m * ldc + n]);
        }
    }
}

// Helper function for approximate GELU activation
static float gelu_approx_activation(float x) {
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3.0f))));
}

// Compute with GELU activation
void xdnn_sgemm_f32nf4f32_compute_gelu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc) {
    DEBUG_PRINT();
    
    // First compute the regular matrix multiplication
    xdnn_sgemm_f32nf4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Then apply GELU activation
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * ldc + n] = gelu_approx_activation(C[m * ldc + n]);
        }
    }
}

// Compute with bias addition
void xdnn_sgemm_f32nf4f32_compute_biasadd(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias) {
    DEBUG_PRINT();
    
    // First compute the regular matrix multiplication
    xdnn_sgemm_f32nf4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Then add bias
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * ldc + n] += bias[n];
        }
    }
}

// Helper function for ReLU activation
static float relu_activation(float x) {
    return std::max(0.0f, x);
}

// Compute with bias addition and ReLU activation
void xdnn_sgemm_f32nf4f32_compute_biasadd_relu(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias) {
    DEBUG_PRINT();
    
    // First compute with bias addition
    xdnn_sgemm_f32nf4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    
    // Then apply ReLU activation
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * ldc + n] = relu_activation(C[m * ldc + n]);
        }
    }
}

// Compute with residual addition
void xdnn_sgemm_f32nf4f32_compute_residential(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias, const float *res, int ldres) {
    DEBUG_PRINT();
    
    // First compute with bias addition
    xdnn_sgemm_f32nf4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    
    // Then add residual
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * ldc + n] += res[m * ldres + n];
        }
    }
}

// Extended residual computation with scaling
void xdnn_sgemm_f32nf4f32_compute_resext(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *bias, 
        float gamma, const float *res, int ldres) {
    DEBUG_PRINT();
    
    // First compute with bias addition
    xdnn_sgemm_f32nf4f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
    
    // Then add scaled residual
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * ldc + n] += gamma * res[m * ldres + n];
        }
    }
}

// Residual multiplication
void xdnn_sgemm_f32nf4f32_compute_resmul(bool transA, int M, int N, int K,
        float alpha, const float *A, int lda, const XDNN_NF4x2 *packedB, const float *scaleB, const float *zeroB,
        float beta, float *C, int ldc, const float *res, int ldres) {
    DEBUG_PRINT();
    
    // First compute the regular matrix multiplication
    xdnn_sgemm_f32nf4f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
    
    // Then multiply by residual
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * ldc + n] *= res[m * ldres + n];
        }
    }
}

// Small SGEMM implementation for single-threaded small matrices
void small_sgemm_f32nf4f32(int M, int N, int K, const float *A, int lda,
        const XDNN_NF4x2 *B, int ldb, const float *scaleB, const float *zeroB, float *C, int ldc) {
    DEBUG_PRINT();
    
    // Implementation is similar to regular compute but optimized for small sizes
    // For now, we'll use the same implementation as the regular compute function
    xdnn_sgemm_f32nf4f32_compute(false, M, N, K, 1.0f, A, lda, B, scaleB, zeroB, 0.0f, C, ldc);
}

} // extern "C"