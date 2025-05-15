#include "conversion.h"
#include "softmax.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <immintrin.h>
#include "conversion.h"

// Implementation of softmax functions for float (F32)
void small_softmax_f32(float *data, const float scale, int size) {
    // Find max value
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        max_val = std::max(max_val, data[i]);
    }
    
    // Compute exp(x_i - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        data[i] = std::exp((data[i] - max_val) * scale);
        sum += data[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        data[i] *= inv_sum;
    }
}

// Implementation of softmax functions for BF16
void small_softmax_bf16(XDNN_BF16 *data, const float scale, int size) {
    // Convert BF16 to FP32 for computation
    // Find max value
    float max_val = static_cast<float>(data[0]);
    for (int i = 1; i < size; i++) {
        max_val = std::max(max_val, static_cast<float>(data[i]));
    }
    
    // Compute exp(x_i - max) and sum
    float sum = 0.0f;
    std::vector<float> temp(size);
    
    for (int i = 0; i < size; i++) {
        temp[i] = std::exp((static_cast<float>(data[i]) - max_val) * scale);
        sum += temp[i];
    }
    
    // Normalize and convert back to BF16
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        data[i] = temp[i] * inv_sum; // implicit conversion to BF16
    }
}
