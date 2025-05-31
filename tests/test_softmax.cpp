#include "gtest/gtest.h"
#include "softmax.h" // Should be found via include_directories
#include "data_types/bfloat16.h"
#include "test_common.h" // For BF16_PRECISION_TOLERANCE

#include <vector>
#include <numeric>
#include <cmath>
#include <cstdlib> // For std::rand and std::srand
#include <algorithm> // For std::transform with XDNN_BF16 if needed, and std::max_element

// Helper function for golden softmax calculation (float)
// Calculates softmax(input[i] * scale)
std::vector<float> calculate_golden_softmax(const std::vector<float>& input_values, float scale_param, int size) {
    std::vector<float> result(size);
    if (size == 0) {
        return result;
    }

    std::vector<float> scaled_values(size);
    for (int i = 0; i < size; ++i) {
        scaled_values[i] = input_values[i] * scale_param;
    }

    // Subtract max for numerical stability
    float max_val = scaled_values[0];
    for (int i = 1; i < size; ++i) {
        if (scaled_values[i] > max_val) {
            max_val = scaled_values[i];
        }
    }

    float sum_exp_values = 0.0f;
    for (int i = 0; i < size; ++i) {
        result[i] = std::exp(scaled_values[i] - max_val);
        sum_exp_values += result[i];
    }

    if (sum_exp_values > 1e-9f) { // Avoid division by zero or very small numbers
        for (int i = 0; i < size; ++i) {
            result[i] /= sum_exp_values;
        }
    } else {
        // If sum_exp is zero (e.g., all inputs very negative, or scale is such that all products are very negative)
        // or if all inputs are identical and scale is 0 (exp(0)=1, sum=size), this branch might be hit if 1e-9f is too large.
        // A uniform distribution is a common fallback.
        if (size > 0) {
            for (int i = 0; i < size; ++i) {
                result[i] = 1.0f / static_cast<float>(size);
            }
        }
    }
    return result;
}

// Test fixture for Softmax tests
class SoftmaxTest : public ::testing::Test {};

TEST_F(SoftmaxTest, SmallSoftmaxF32_Basic) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    float scale = 1.0f;
    int size = data.size();
    std::vector<float> expected = calculate_golden_softmax(data, scale, size);
    
    std::vector<float> actual_data = data; // Copy because small_softmax_f32 modifies in-place
    small_softmax_f32(actual_data.data(), scale, size);

    ASSERT_EQ(actual_data.size(), expected.size());
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(actual_data[i], expected[i], 1e-5f);
    }
}

TEST_F(SoftmaxTest, SmallSoftmaxF32_DifferentScale) {
    std::vector<float> data = {1.0f, 0.5f, 0.2f};
    float scale = 2.0f;
    int size = data.size();
    std::vector<float> expected = calculate_golden_softmax(data, scale, size);

    std::vector<float> actual_data = data;
    small_softmax_f32(actual_data.data(), scale, size);

    ASSERT_EQ(actual_data.size(), expected.size());
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(actual_data[i], expected[i], 1e-5f);
    }
}

TEST_F(SoftmaxTest, SmallSoftmaxF32_NegativeValues) {
    std::vector<float> data = {-1.0f, -2.0f, -0.5f};
    float scale = 1.0f;
    int size = data.size();
    std::vector<float> expected = calculate_golden_softmax(data, scale, size);

    std::vector<float> actual_data = data;
    small_softmax_f32(actual_data.data(), scale, size);

    ASSERT_EQ(actual_data.size(), expected.size());
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(actual_data[i], expected[i], 1e-5f);
    }
}

TEST_F(SoftmaxTest, SmallSoftmaxF32_AllSameValues) {
    std::vector<float> data = {2.0f, 2.0f, 2.0f, 2.0f};
    float scale = 1.0f;
    int size = data.size();
    std::vector<float> expected = calculate_golden_softmax(data, scale, size);

    std::vector<float> actual_data = data;
    small_softmax_f32(actual_data.data(), scale, size);

    ASSERT_EQ(actual_data.size(), expected.size());
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(actual_data[i], 1.0f / size, 1e-5f); // More direct check for this case
        EXPECT_NEAR(actual_data[i], expected[i], 1e-5f);
    }
}

TEST_F(SoftmaxTest, SmallSoftmaxF32_SingleValue) {
    std::vector<float> data = {5.0f};
    float scale = 1.0f;
    int size = data.size();
    std::vector<float> expected = calculate_golden_softmax(data, scale, size); 

    std::vector<float> actual_data = data;
    small_softmax_f32(actual_data.data(), scale, size);

    ASSERT_EQ(actual_data.size(), expected.size());
    ASSERT_EQ(size, 1);
    EXPECT_NEAR(actual_data[0], 1.0f, 1e-5f); // Single element softmax is 1
    EXPECT_NEAR(actual_data[0], expected[0], 1e-5f);
}

TEST_F(SoftmaxTest, SmallSoftmaxF32_ZeroScale) {
    std::vector<float> data = {1.0f, 10.0f, -5.0f, 100.0f};
    float scale = 0.0f;
    int size = data.size();
    std::vector<float> expected = calculate_golden_softmax(data, scale, size); 

    std::vector<float> actual_data = data;
    small_softmax_f32(actual_data.data(), scale, size);

    ASSERT_EQ(actual_data.size(), expected.size());
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(actual_data[i], 1.0f / size, 1e-5f); // More direct check
        EXPECT_NEAR(actual_data[i], expected[i], 1e-5f);
    }
}

TEST_F(SoftmaxTest, SmallSoftmaxF32_RandomDataVariableSizes) {
    float scale = 0.088388f;
    
    // Seed for reproducible random data
    std::srand(42);
    
    for (int size = 1025; size <= 2047; ++size) {
        // Generate random float data
        std::vector<float> data_f32(size);
        for (int i = 0; i < size; ++i) {
            // Generate random values between -10.0 and 10.0
            data_f32[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 20.0f - 10.0f;
        }
        
        // Calculate expected result using golden softmax
        std::vector<float> expected_f32 = calculate_golden_softmax(data_f32, scale, size);
        
        // Run the f32 softmax function
        std::vector<float> actual_data_f32 = data_f32;
        small_softmax_f32(actual_data_f32.data(), scale, size);
        
        // Verify results
        ASSERT_EQ(actual_data_f32.size(), expected_f32.size()) << "Size mismatch for size=" << size;
        
        for (int i = 0; i < size; ++i) {
            EXPECT_NEAR(actual_data_f32[i], expected_f32[i], 1e-5f)
                << "Mismatch at index " << i << " for size=" << size;
        }
        
        // Verify that the sum is approximately 1.0 (basic softmax property)
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum += actual_data_f32[i];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Sum not close to 1.0 for size=" << size;
    }
}

// Tests for small_softmax_bf16
TEST_F(SoftmaxTest, SmallSoftmaxBf16_Basic) {
    std::vector<float> data_f32 = {1.0f, 2.0f, 3.0f};
    float scale = 1.0f;
    int size = data_f32.size();

    std::vector<XDNN_BF16> data_bf16(size);
    for(int i=0; i<size; ++i) data_bf16[i] = XDNN_BF16(data_f32[i]);

    std::vector<float> expected_f32 = calculate_golden_softmax(data_f32, scale, size);
    
    std::vector<XDNN_BF16> actual_data_bf16 = data_bf16; 
    small_softmax_bf16(actual_data_bf16.data(), scale, size);

    ASSERT_EQ(actual_data_bf16.size(), expected_f32.size());
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(static_cast<float>(actual_data_bf16[i]), expected_f32[i], BF16_PRECISION_TOLERANCE);
    }
}

TEST_F(SoftmaxTest, SmallSoftmaxBf16_DifferentScale) {
    std::vector<float> data_f32 = {1.0f, 0.5f, 0.2f};
    float scale = 0.5f;
    int size = data_f32.size();

    std::vector<XDNN_BF16> data_bf16(size);
    for(int i=0; i<size; ++i) data_bf16[i] = XDNN_BF16(data_f32[i]);

    std::vector<float> expected_f32 = calculate_golden_softmax(data_f32, scale, size);
    
    std::vector<XDNN_BF16> actual_data_bf16 = data_bf16;
    small_softmax_bf16(actual_data_bf16.data(), scale, size);

    ASSERT_EQ(actual_data_bf16.size(), expected_f32.size());
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(static_cast<float>(actual_data_bf16[i]), expected_f32[i], BF16_PRECISION_TOLERANCE);
    }
}

TEST_F(SoftmaxTest, SmallSoftmaxBf16_LargeValues) { 
    std::vector<float> data_f32 = {10.0f, 20.0f, 15.0f}; // Values that might stress bf16 precision
    float scale = 1.0f;
    int size = data_f32.size();

    std::vector<XDNN_BF16> data_bf16(size);
    for(int i=0; i<size; ++i) data_bf16[i] = XDNN_BF16(data_f32[i]);

    std::vector<float> expected_f32 = calculate_golden_softmax(data_f32, scale, size);
    
    std::vector<XDNN_BF16> actual_data_bf16 = data_bf16;
    small_softmax_bf16(actual_data_bf16.data(), scale, size);

    ASSERT_EQ(actual_data_bf16.size(), expected_f32.size());
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(static_cast<float>(actual_data_bf16[i]), expected_f32[i], BF16_PRECISION_TOLERANCE);
    }
}

TEST_F(SoftmaxTest, SmallSoftmaxBf16_ZeroScale) {
    std::vector<float> data_f32 = {10.0f, 20.0f, 15.0f, 50.0f};
    float scale = 0.0f;
    int size = data_f32.size();

    std::vector<XDNN_BF16> data_bf16(size);
    for(int i=0; i<size; ++i) data_bf16[i] = XDNN_BF16(data_f32[i]);

    std::vector<float> expected_f32 = calculate_golden_softmax(data_f32, scale, size);
    
    std::vector<XDNN_BF16> actual_data_bf16 = data_bf16;
    small_softmax_bf16(actual_data_bf16.data(), scale, size);

    ASSERT_EQ(actual_data_bf16.size(), expected_f32.size());
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(static_cast<float>(actual_data_bf16[i]), 1.0f / size, BF16_PRECISION_TOLERANCE);
        EXPECT_NEAR(static_cast<float>(actual_data_bf16[i]), expected_f32[i], BF16_PRECISION_TOLERANCE);
    }
}

TEST_F(SoftmaxTest, SmallSoftmaxBf16_SingleValue) {
    std::vector<float> data_f32 = {12.3f};
    float scale = 0.75f;
    int size = data_f32.size();

    std::vector<XDNN_BF16> data_bf16(size);
    for(int i=0; i<size; ++i) data_bf16[i] = XDNN_BF16(data_f32[i]);

    std::vector<float> expected_f32 = calculate_golden_softmax(data_f32, scale, size);
    
    std::vector<XDNN_BF16> actual_data_bf16 = data_bf16;
    small_softmax_bf16(actual_data_bf16.data(), scale, size);

    ASSERT_EQ(actual_data_bf16.size(), expected_f32.size());
    ASSERT_EQ(size,1);
    EXPECT_NEAR(static_cast<float>(actual_data_bf16[0]), 1.0f, BF16_PRECISION_TOLERANCE);
    EXPECT_NEAR(static_cast<float>(actual_data_bf16[0]), expected_f32[0], BF16_PRECISION_TOLERANCE);
}

TEST_F(SoftmaxTest, SmallSoftmaxBf16_RandomDataVariableSizes) {
    float scale = 0.088388f;
    
    // Seed for reproducible random data
    std::srand(42);
    
    for (int size = 1; size <= 1024; ++size) {
        // Generate random float data
        std::vector<float> data_f32(size);
        for (int i = 0; i < size; ++i) {
            // Generate random values between -10.0 and 10.0
            data_f32[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 20.0f - 10.0f;
        }
        
        // Convert to bf16
        std::vector<XDNN_BF16> data_bf16(size);
        for (int i = 0; i < size; ++i) {
            data_bf16[i] = XDNN_BF16(data_f32[i]);
        }
        
        // Calculate expected result using golden softmax
        std::vector<float> expected_f32 = calculate_golden_softmax(data_f32, scale, size);
        
        // Run the bf16 softmax function
        std::vector<XDNN_BF16> actual_data_bf16 = data_bf16;
        small_softmax_bf16(actual_data_bf16.data(), scale, size);
        
        // Verify results
        ASSERT_EQ(actual_data_bf16.size(), expected_f32.size()) << "Size mismatch for size=" << size;
        
        for (int i = 0; i < size; ++i) {
            EXPECT_NEAR(static_cast<float>(actual_data_bf16[i]), expected_f32[i], BF16_PRECISION_TOLERANCE)
                << "Mismatch at index " << i << " for size=" << size;
        }
        
        // Verify that the sum is approximately 1.0 (basic softmax property)
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum += static_cast<float>(actual_data_bf16[i]);
        }
        EXPECT_NEAR(sum, 1.0f, BF16_PRECISION_TOLERANCE) << "Sum not close to 1.0 for size=" << size;
    }
}
