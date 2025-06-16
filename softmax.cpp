#include "conversion.h"
#include "softmax.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <immintrin.h>
#include "conversion.h"
#include "debug_print.h"

// Check for AVX-512 support
#ifdef __AVX512F__
#define HAS_AVX512
#endif

// Fast approximation of exp using AVX-512
#ifdef HAS_AVX512
inline __m512 fast_exp_ps(__m512 x) {
    // Clamp input to reasonable range to avoid overflow
    const __m512 max_input = _mm512_set1_ps(88.0f);
    const __m512 min_input = _mm512_set1_ps(-88.0f);
    x = _mm512_max_ps(_mm512_min_ps(x, max_input), min_input);
    
    // For now, fall back to scalar exp per element
    alignas(64) float temp_values[16];
    _mm512_store_ps(temp_values, x);
    for (int i = 0; i < 16; i++) {
        temp_values[i] = std::exp(temp_values[i]);
    }
    return _mm512_load_ps(temp_values);
}
#endif

// AVX-512 optimized implementation of softmax for float (F32)
void small_softmax_f32(float *data, const float scale, int size) {
    DEBUG_PRINT();
    // DEBUG_PRINT_PARAMS("scale = %f, size = %d\n", scale, size);
    
#ifdef HAS_AVX512
    constexpr int simd_width = 16; // AVX-512 processes 16 floats at once
    const int vectorized_size = (size / simd_width) * simd_width;
    
    // Find max value using AVX-512
    __m512 max_vec = _mm512_set1_ps(-INFINITY);
    
    // Vectorized max finding
    for (int i = 0; i < vectorized_size; i += simd_width) {
        __m512 data_vec = _mm512_loadu_ps(&data[i]);
        max_vec = _mm512_max_ps(max_vec, data_vec);
    }
    
    // Horizontal max reduction
    float max_val = _mm512_reduce_max_ps(max_vec);
    
    // Handle remaining elements
    for (int i = vectorized_size; i < size; i++) {
        max_val = std::max(max_val, data[i]);
    }
    
    // Broadcast scale and max values
    const __m512 scale_vec = _mm512_set1_ps(scale);
    const __m512 max_vec_broadcast = _mm512_set1_ps(max_val);
    
    // Compute exp(x_i - max) * scale and sum using AVX-512
    __m512 sum_vec = _mm512_setzero_ps();
    
    for (int i = 0; i < vectorized_size; i += simd_width) {
        __m512 data_vec = _mm512_loadu_ps(&data[i]);
        // (data[i] - max_val) * scale
        __m512 scaled_vec = _mm512_mul_ps(_mm512_sub_ps(data_vec, max_vec_broadcast), scale_vec);
        // exp(scaled_vec) using fast approximation
        __m512 exp_vec = fast_exp_ps(scaled_vec);
        _mm512_storeu_ps(&data[i], exp_vec);
        sum_vec = _mm512_add_ps(sum_vec, exp_vec);
    }
    
    // Horizontal sum reduction
    float sum = _mm512_reduce_add_ps(sum_vec);
    
    // Handle remaining elements
    for (int i = vectorized_size; i < size; i++) {
        data[i] = std::exp((data[i] - max_val) * scale);
        sum += data[i];
    }
    
    // Normalize using AVX-512
    const __m512 inv_sum_vec = _mm512_set1_ps(1.0f / sum);
    
    for (int i = 0; i < vectorized_size; i += simd_width) {
        __m512 data_vec = _mm512_loadu_ps(&data[i]);
        __m512 result = _mm512_mul_ps(data_vec, inv_sum_vec);
        _mm512_storeu_ps(&data[i], result);
    }
    
    // Handle remaining elements
    const float inv_sum = 1.0f / sum;
    for (int i = vectorized_size; i < size; i++) {
        data[i] *= inv_sum;
    }
#else
    // Fallback implementation using AVX2 or scalar operations
    constexpr int simd_width = 8; // AVX2 processes 8 floats at once
    const int vectorized_size = (size / simd_width) * simd_width;
    
    // Find max value using AVX2
    __m256 max_vec = _mm256_set1_ps(-INFINITY);
    
    for (int i = 0; i < vectorized_size; i += simd_width) {
        __m256 data_vec = _mm256_loadu_ps(&data[i]);
        max_vec = _mm256_max_ps(max_vec, data_vec);
    }
    
    // Extract max from AVX2 register
    alignas(32) float max_array[8];
    _mm256_store_ps(max_array, max_vec);
    float max_val = max_array[0];
    for (int i = 1; i < 8; i++) {
        max_val = std::max(max_val, max_array[i]);
    }
    
    // Handle remaining elements
    for (int i = vectorized_size; i < size; i++) {
        max_val = std::max(max_val, data[i]);
    }
    
    // Compute exp and sum
    const __m256 scale_vec = _mm256_set1_ps(scale);
    const __m256 max_vec_broadcast = _mm256_set1_ps(max_val);
    __m256 sum_vec = _mm256_setzero_ps();
    
    for (int i = 0; i < vectorized_size; i += simd_width) {
        __m256 data_vec = _mm256_loadu_ps(&data[i]);
        __m256 scaled_vec = _mm256_mul_ps(_mm256_sub_ps(data_vec, max_vec_broadcast), scale_vec);
        
        // Manual exp approximation for AVX2 (or use libm)
        alignas(32) float temp_exp[8];
        _mm256_store_ps(temp_exp, scaled_vec);
        for (int j = 0; j < 8; j++) {
            temp_exp[j] = std::exp(temp_exp[j]);
        }
        __m256 exp_vec = _mm256_load_ps(temp_exp);
        
        _mm256_storeu_ps(&data[i], exp_vec);
        sum_vec = _mm256_add_ps(sum_vec, exp_vec);
    }
    
    // Extract sum from AVX2 register
    alignas(32) float sum_array[8];
    _mm256_store_ps(sum_array, sum_vec);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        sum += sum_array[i];
    }
    
    // Handle remaining elements
    for (int i = vectorized_size; i < size; i++) {
        data[i] = std::exp((data[i] - max_val) * scale);
        sum += data[i];
    }
    
    // Normalize
    const __m256 inv_sum_vec = _mm256_set1_ps(1.0f / sum);
    for (int i = 0; i < vectorized_size; i += simd_width) {
        __m256 data_vec = _mm256_loadu_ps(&data[i]);
        __m256 result = _mm256_mul_ps(data_vec, inv_sum_vec);
        _mm256_storeu_ps(&data[i], result);
    }
    
    const float inv_sum = 1.0f / sum;
    for (int i = vectorized_size; i < size; i++) {
        data[i] *= inv_sum;
    }
#endif
}

// AVX-512 optimized implementation of softmax for BF16
void small_softmax_bf16(XDNN_BF16 *data, const float scale, int size) {
    DEBUG_PRINT();
    // DEBUG_PRINT_PARAMS("scale = %f, size = %d\n", scale, size);
    
#ifdef HAS_AVX512
    constexpr int simd_width = 16; // AVX-512 processes 16 floats at once
    const int vectorized_size = (size / simd_width) * simd_width;
    
    // Manual BF16 to FP32 conversion and max finding
    __m512 max_vec = _mm512_set1_ps(-INFINITY);
    
    // Vectorized BF16 to FP32 conversion and max finding
    for (int i = 0; i < vectorized_size; i += simd_width) {
        // Manual BF16 to FP32 conversion
        alignas(64) float fp32_values[16];
        for (int j = 0; j < 16; j++) {
            fp32_values[j] = static_cast<float>(data[i + j]);
        }
        
        __m512 fp32_vec = _mm512_load_ps(fp32_values);
        max_vec = _mm512_max_ps(max_vec, fp32_vec);
    }
    
    // Horizontal max reduction
    float max_val = _mm512_reduce_max_ps(max_vec);
    
    // Handle remaining elements
    for (int i = vectorized_size; i < size; i++) {
        max_val = std::max(max_val, static_cast<float>(data[i]));
    }
    
    // Broadcast scale and max values
    const __m512 scale_vec = _mm512_set1_ps(scale);
    const __m512 max_vec_broadcast = _mm512_set1_ps(max_val);
    
    // Allocate temporary storage for FP32 values
    alignas(64) float temp_fp32[vectorized_size];
    
    // Compute exp(x_i - max) * scale and sum using AVX-512
    __m512 sum_vec = _mm512_setzero_ps();
    
    for (int i = 0; i < vectorized_size; i += simd_width) {
        // Load and convert BF16 to FP32 manually
        alignas(64) float fp32_values[16];
        for (int j = 0; j < 16; j++) {
            fp32_values[j] = static_cast<float>(data[i + j]);
        }
        __m512 fp32_vec = _mm512_load_ps(fp32_values);
        
        // (data[i] - max_val) * scale
        __m512 scaled_vec = _mm512_mul_ps(_mm512_sub_ps(fp32_vec, max_vec_broadcast), scale_vec);
        
        // exp(scaled_vec) using fast approximation
        __m512 exp_vec = fast_exp_ps(scaled_vec);
        
        // Store temporarily in FP32 format
        _mm512_store_ps(&temp_fp32[i], exp_vec);
        
        sum_vec = _mm512_add_ps(sum_vec, exp_vec);
    }
    
    // Horizontal sum reduction
    float sum = _mm512_reduce_add_ps(sum_vec);
    
    // Handle remaining elements (non-vectorized)
    std::vector<float> temp_remaining;
    if (vectorized_size < size) {
        temp_remaining.resize(size - vectorized_size);
        for (int i = vectorized_size; i < size; i++) {
            float val = std::exp((static_cast<float>(data[i]) - max_val) * scale);
            temp_remaining[i - vectorized_size] = val;
            sum += val;
        }
    }
    
    // Normalize using AVX-512 and convert back to BF16
    const __m512 inv_sum_vec = _mm512_set1_ps(1.0f / sum);
    
    for (int i = 0; i < vectorized_size; i += simd_width) {
        __m512 fp32_vec = _mm512_load_ps(&temp_fp32[i]);
        __m512 normalized = _mm512_mul_ps(fp32_vec, inv_sum_vec);
        
        // Manual FP32 to BF16 conversion
        alignas(64) float result_fp32[16];
        _mm512_store_ps(result_fp32, normalized);
        for (int j = 0; j < 16; j++) {
            data[i + j] = result_fp32[j]; // Implicit conversion to BF16
        }
    }
    
    // Handle remaining elements
    const float inv_sum = 1.0f / sum;
    for (int i = vectorized_size; i < size; i++) {
        data[i] = temp_remaining[i - vectorized_size] * inv_sum;
    }
#else
    // Fallback implementation for systems without AVX-512
    // Find max value
    float max_val = static_cast<float>(data[0]);
    for (int i = 1; i < size; i++) {
        max_val = std::max(max_val, static_cast<float>(data[i]));
    }
    
    // Compute exp(x_i - max) and sum with vectorized operations when possible
    constexpr int simd_width = 8; // AVX2 processes 8 floats
    const int vectorized_size = (size / simd_width) * simd_width;
    
    std::vector<float> temp(size);
    
    // Vectorized processing for majority of elements
    const __m256 scale_vec = _mm256_set1_ps(scale);
    const __m256 max_vec = _mm256_set1_ps(max_val);
    __m256 sum_vec = _mm256_setzero_ps();
    
    for (int i = 0; i < vectorized_size; i += simd_width) {
        // Convert BF16 to FP32 manually
        alignas(32) float fp32_values[8];
        for (int j = 0; j < 8; j++) {
            fp32_values[j] = static_cast<float>(data[i + j]);
        }
        
        __m256 fp32_vec = _mm256_load_ps(fp32_values);
        __m256 scaled_vec = _mm256_mul_ps(_mm256_sub_ps(fp32_vec, max_vec), scale_vec);
        
        // Compute exp manually since AVX2 doesn't have native exp
        alignas(32) float exp_values[8];
        _mm256_store_ps(exp_values, scaled_vec);
        for (int j = 0; j < 8; j++) {
            exp_values[j] = std::exp(exp_values[j]);
            temp[i + j] = exp_values[j];
        }
        
        __m256 exp_vec = _mm256_load_ps(exp_values);
        sum_vec = _mm256_add_ps(sum_vec, exp_vec);
    }
    
    // Extract sum from AVX2 register
    alignas(32) float sum_array[8];
    _mm256_store_ps(sum_array, sum_vec);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        sum += sum_array[i];
    }
    
    // Handle remaining elements
    for (int i = vectorized_size; i < size; i++) {
        temp[i] = std::exp((static_cast<float>(data[i]) - max_val) * scale);
        sum += temp[i];
    }
    
    // Normalize and convert back to BF16
    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        data[i] = temp[i] * inv_sum; // implicit conversion to BF16
    }
#endif
}
