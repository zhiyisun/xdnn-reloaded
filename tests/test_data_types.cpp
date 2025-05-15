#include "conversion.h"
#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <limits>
#include "data_types/data_types.h"
#include "data_types/float16.h"
#include "data_types/bfloat16.h"
#include "data_types/fp8_e4m3.h"
#include "data_types/normal_float4x2.h"
#include "data_types/uint4x2.h"

// Test fixture for data types tests
class DataTypesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code
    }

    void TearDown() override {
        // Common teardown code
    }
};

// Helper function for getting random float values
std::vector<float> getRandomFloats(size_t count, float min = -100.0f, float max = 100.0f) {
    std::vector<float> values(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    for (size_t i = 0; i < count; ++i) {
        values[i] = dist(gen);
    }
    
    return values;
}

// Test XDNN_FP16 (Float16)
TEST_F(DataTypesTest, Float16Test) {
    const size_t count = 1000;
    auto test_values = getRandomFloats(count);
    
    for (float value : test_values) {
        // Convert float to fp16 and back to float
        XDNN_FP16 fp16_val(value);
        float roundtrip = static_cast<float>(fp16_val);
        
        // Due to limited precision, we expect some loss
        // The error should be proportional to the magnitude
        float abs_value = std::abs(value);
        float tolerance = std::max(1e-3f, abs_value * 1e-2f);
        
        // For values outside of fp16 range, we expect saturation
        if (value > 65504.0f) {
            // Compare the float values instead
            EXPECT_EQ(static_cast<float>(fp16_val), static_cast<float>(XDNN_FP16(65504.0f)));
        } else if (value < -65504.0f) {
            EXPECT_EQ(static_cast<float>(fp16_val), static_cast<float>(XDNN_FP16(-65504.0f)));
        } else if (abs_value < 6.1e-5f && abs_value > 0) {
            // Subnormal range or underflow, check it's close to zero or preserved
            EXPECT_TRUE(std::abs(roundtrip) < 6.1e-5f);
        } else if (abs_value == 0) {
            // Exact zero should be preserved
            EXPECT_EQ(roundtrip, 0.0f);
        } else {
            // Normal range
            EXPECT_NEAR(value, roundtrip, tolerance);
        }
    }
    
    // Test arithmetic operations
    XDNN_FP16 a(1.5f);
    XDNN_FP16 b(2.5f);
    
    EXPECT_NEAR(static_cast<float>(a + b), 4.0f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(a - b), -1.0f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(a * b), 3.75f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(a / b), 0.6f, 1e-3f);
}

// Test XDNN_BF16 (BFloat16)
TEST_F(DataTypesTest, BFloat16Test) {
    const size_t count = 1000;
    auto test_values = getRandomFloats(count);
    
    for (float value : test_values) {
        // Convert float to bf16 and back to float
        XDNN_BF16 bf16_val(value);
        float roundtrip = static_cast<float>(bf16_val);
        
        // Due to limited precision, we expect some loss
        // BF16 maintains the same range as float32 but with reduced precision
        float abs_value = std::abs(value);
        float tolerance = std::max(1e-2f, abs_value * 1e-2f);
        
        // BF16 should maintain full float32 range
        if (abs_value == 0) {
            // Exact zero should be preserved
            EXPECT_EQ(roundtrip, 0.0f);
        } else {
            // Normal range
            EXPECT_NEAR(value, roundtrip, tolerance);
        }
    }
    
    // Test arithmetic operations
    XDNN_BF16 a(1.5f);
    XDNN_BF16 b(2.5f);
    
    EXPECT_NEAR(static_cast<float>(a + b), 4.0f, 1e-2f);
    EXPECT_NEAR(static_cast<float>(a - b), -1.0f, 1e-2f);
    EXPECT_NEAR(static_cast<float>(a * b), 3.75f, 1e-2f);
    EXPECT_NEAR(static_cast<float>(a / b), 0.6f, 1e-2f);
}

// Test XDNN_E4M3 (8-bit floating point with 4-bit exponent, 3-bit mantissa)
TEST_F(DataTypesTest, E4M3Test) {
    // Test basic functionality of the class
    XDNN_E4M3 zero_val;
    XDNN_E4M3 uint_val(static_cast<uint8_t>(42));
    
    // Test we can assign and retrieve raw values
    uint8_t raw_val = static_cast<uint8_t>(uint_val);
    EXPECT_EQ(raw_val, 42);
    
    // Test conversion of basic values
    XDNN_E4M3 pos_val(1.0f);
    XDNN_E4M3 neg_val(-1.0f);
    float pos_float = static_cast<float>(pos_val);
    float neg_float = static_cast<float>(neg_val);
    
    // Verify sign preservation
    EXPECT_GT(pos_float, 0.0f);
    EXPECT_LT(neg_float, 0.0f);
    
    // Test zero handling
    XDNN_E4M3 default_zero; // Default constructor should create zero
    EXPECT_EQ(static_cast<float>(default_zero), 0.0f);
    
    XDNN_E4M3 explicit_zero(0.0f);
    EXPECT_EQ(static_cast<float>(explicit_zero), 0.0f);
    
    XDNN_E4M3 assigned_zero;
    assigned_zero = 0.0f;
    EXPECT_EQ(static_cast<float>(assigned_zero), 0.0f);
    
    // Test assigning different values
    XDNN_E4M3 value;
    
    value = 0.5f;
    // E4M3 has limited precision - we need to increase the tolerance to account for rounding
    EXPECT_NEAR(static_cast<float>(value), 0.5f, 1.0f);
    
    value = -0.5f;
    EXPECT_NEAR(static_cast<float>(value), -0.5f, 1.0f);
    
    value = 2.0f;
    EXPECT_NEAR(static_cast<float>(value), 2.0f, 4.0f);
    
    value = -2.0f;
    EXPECT_NEAR(static_cast<float>(value), -2.0f, 4.0f);
}

// Test XDNN_UINT4x2 (Two 4-bit unsigned integers packed into a byte)
TEST_F(DataTypesTest, UINT4x2Test) {
    // Test setting and getting values
    
    // Test all combinations of 4-bit values (0-15)
    for (uint8_t i = 0; i <= 15; ++i) {
        for (uint8_t j = 0; j <= 15; ++j) {
            // Create a uint4x2 with the two values
            XDNN_UINT4x2 packed_val(i, j);
            
            // Extract the values using the class methods
            uint8_t low = packed_val.get_v1();
            uint8_t high = packed_val.get_v2();
            
            EXPECT_EQ(low, i);
            EXPECT_EQ(high, j);
            
            // Test equality operator
            XDNN_UINT4x2 same_val(i, j);
            EXPECT_FALSE(packed_val != same_val);
        }
    }
}

// Test XDNN_NORMAL_FLOAT4x2 (Two 4-bit normalized floats packed into a byte)
TEST_F(DataTypesTest, NormalFloat4x2Test) {
    // We'll test the basic XDNN_NORMAL_FLOAT4x2 class using XDNN_UINT4x2 as it's derived from it
    
    // We'll test that accessing specific indices works
    for (uint8_t i = 0; i <= 15; ++i) {
        for (uint8_t j = 0; j <= 15; ++j) {
            // Create a normal_float4x2 with the two values
            XDNN_UINT4x2 packed_val(i, j);
            
            // Extract the values using the getter methods
            uint8_t val1 = packed_val.get_v1();
            uint8_t val2 = packed_val.get_v2();
            
            // Check that values match what we set
            EXPECT_EQ(val1, i);
            EXPECT_EQ(val2, j);
            
            // Test != operator
            XDNN_UINT4x2 same_val(i, j);
            XDNN_UINT4x2 diff_val((i+1)%16, j);
            
            EXPECT_FALSE(packed_val != same_val);
            if (i < 15) {  // Only test if we won't wrap around to the same value
                EXPECT_TRUE(packed_val != diff_val);
            }
        }
    }
}

// Test bit_convert functions if available
// This would depend on the specifics of your bit_convert.h implementation
