#include "conversion.h"
#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <vector>
#include "softmax.h"
#include "data_types/bfloat16.h"
#include "../platform_detection.h"

// Test fixture for softmax tests
class SoftmaxTest : public ::testing::Test {
protected:
    xdnn::CPUFeatures cpuFeatures;
    
    void SetUp() override {
        // Detect CPU features for conditional tests
        cpuFeatures = xdnn::detectCPUFeatures();
    }

    void TearDown() override {
        // Common teardown code
    }
    
    // Skip if not running on Intel Xeon with AVX512
    void SkipIfNotXeon() {
        bool isXeon = cpuFeatures.supportsAVX512();
        if (!isXeon) {
            GTEST_SKIP() << "Skipping test that requires Intel Xeon platform which is not available";
        }
    }
    
    // Skip if hardware doesn't support BF16 operations
    void SkipIfBF16NotSupported() {
        // BF16 operations require AVX512 support
        bool supportsBF16 = cpuFeatures.supportsAVX512();
        if (!supportsBF16) {
            GTEST_SKIP() << "Skipping test that requires BF16 support which is not available on this CPU";
        }
    }
    
    // Reference softmax implementation for validation
    void reference_softmax(float* data, const float scale, int size) {
        // Find max for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < size; ++i) {
            max_val = std::max(max_val, data[i]);
        }
        
        // Compute exp(x * scale - max) for each element
        std::vector<float> exp_values(size);
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            exp_values[i] = std::exp(scale * data[i] - max_val);
            sum += exp_values[i];
        }
        
        // Normalize
        for (int i = 0; i < size; ++i) {
            data[i] = exp_values[i] / sum;
        }
    }
    
    // Reference softmax implementation for bfloat16
    void reference_softmax_bf16(XDNN_BF16* data, const float scale, int size) {
        std::vector<float> float_data(size);
        
        // Convert to float
        for (int i = 0; i < size; ++i) {
            float_data[i] = static_cast<float>(data[i]);
        }
        
        // Apply softmax on float values
        reference_softmax(float_data.data(), scale, size);
        
        // Convert back to bfloat16
        for (int i = 0; i < size; ++i) {
            data[i] = XDNN_BF16(float_data[i]);
        }
    }
};

// Test small_softmax_f32 function
TEST_F(SoftmaxTest, F32SoftmaxTest) {
    // Skip if not running on Intel Xeon with optimized code
    SkipIfNotXeon();
    
    // Only continue with test setup if we haven't skipped
    if(::testing::Test::IsSkipped()) {
        return;
    }
    
    std::vector<int> sizes = {1, 8, 16, 32, 64, 128, 256};
    std::vector<float> scales = {1.0f, 2.0f, 0.5f};
    
    for (int size : sizes) {
        for (float scale : scales) {
            // Allocate data
            float* data = new float[size];
            float* reference_data = new float[size];
            
            // Initialize with random values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
            for (int i = 0; i < size; ++i) {
                data[i] = dist(gen);
                reference_data[i] = data[i]; // Copy for reference calculation
            }
            
            // Call the function to test
            small_softmax_f32(data, scale, size);
            
            // Compute reference result
            reference_softmax(reference_data, scale, size);
            
            // Compare results
            const float epsilon = 1e-4f;
            for (int i = 0; i < size; ++i) {
                EXPECT_NEAR(data[i], reference_data[i], epsilon) 
                    << "Mismatch at index " << i << " with size " << size 
                    << " and scale " << scale;
            }
            
            // Clean up
            delete[] data;
            delete[] reference_data;
        }
    }
}

// Test small_softmax_bf16 function
TEST_F(SoftmaxTest, BF16SoftmaxTest) {
    // Skip if BF16 not supported
    SkipIfBF16NotSupported();
    
    // Only continue with test setup if we haven't skipped
    if(::testing::Test::IsSkipped()) {
        return;
    }
    
    std::vector<int> sizes = {1, 8, 16, 32, 64, 128};
    std::vector<float> scales = {1.0f, 2.0f, 0.5f};
    
    for (int size : sizes) {
        for (float scale : scales) {
            // Allocate data
            XDNN_BF16* data = new XDNN_BF16[size];
            XDNN_BF16* reference_data = new XDNN_BF16[size];
            
            // Initialize with random values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
            for (int i = 0; i < size; ++i) {
                data[i] = XDNN_BF16(dist(gen));
                reference_data[i] = data[i]; // Copy for reference calculation
            }
            
            // Call the function to test
            small_softmax_bf16(data, scale, size);
            
            // Compute reference result
            reference_softmax_bf16(reference_data, scale, size);
            
            // Compare results - using higher epsilon due to bf16 precision
            const float epsilon = 1e-2f;
            for (int i = 0; i < size; ++i) {
                EXPECT_NEAR(static_cast<float>(data[i]), 
                            static_cast<float>(reference_data[i]), 
                            epsilon) 
                    << "Mismatch at index " << i << " with size " << size 
                    << " and scale " << scale;
            }
            
            // Clean up
            delete[] data;
            delete[] reference_data;
        }
    }
}

// Edge case tests
TEST_F(SoftmaxTest, EdgeCaseSoftmaxTest) {
    // This test includes both F32 and BF16 cases
    SkipIfNotXeon();
    
    // Only continue with test setup if we haven't skipped
    if(::testing::Test::IsSkipped()) {
        return;
    }
    
    // Test with very large values
    const int size = 8;
    float* large_values = new float[size];
    float* reference_large = new float[size];
    
    for (int i = 0; i < size; ++i) {
        large_values[i] = 1000.0f * i;
        reference_large[i] = large_values[i];
    }
    
    small_softmax_f32(large_values, 1.0f, size);
    reference_softmax(reference_large, 1.0f, size);
    
    // With large values, the largest element should get probability close to 1
    // and all others close to 0
    const float epsilon = 1e-4f;
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(large_values[i], reference_large[i], epsilon);
    }
    
    // Test with uniform values
    float* uniform_values = new float[size];
    float* reference_uniform = new float[size];
    
    for (int i = 0; i < size; ++i) {
        uniform_values[i] = 1.0f;
        reference_uniform[i] = uniform_values[i];
    }
    
    small_softmax_f32(uniform_values, 1.0f, size);
    reference_softmax(reference_uniform, 1.0f, size);
    
    // With uniform values, all probabilities should be equal (1/size)
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(uniform_values[i], 1.0f / size, epsilon);
        EXPECT_NEAR(uniform_values[i], reference_uniform[i], epsilon);
    }
    
    // Clean up
    delete[] large_values;
    delete[] reference_large;
    delete[] uniform_values;
    delete[] reference_uniform;
}
