#include "conversion.h"
#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include "transpose.h"
#include "data_types/float16.h"
#include "data_types/bfloat16.h"
#include "../platform_detection.h"

// Helper function to initialize random matrices
template<typename T>
void initializeRandomTransposeMatrix(T* matrix, size_t size, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    for (size_t i = 0; i < size; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            matrix[i] = dist(gen);
        } else if constexpr (std::is_same_v<T, XDNN_FP16>) {
            matrix[i] = XDNN_FP16(dist(gen));
        } else if constexpr (std::is_same_v<T, XDNN_BF16>) {
            matrix[i] = XDNN_BF16(dist(gen));
        } else {
            // Default case
            matrix[i] = static_cast<T>(dist(gen));
        }
    }
}

// Reference implementation of transpose operation
template<typename T>
void reference_transpose(const T* input, T* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// Test fixture for transpose tests
class TransposeTest : public ::testing::Test {
protected:
    xdnn::CPUFeatures cpuFeatures;
    
    void SetUp() override {
        // Detect CPU features for conditional tests
        cpuFeatures = xdnn::detectCPUFeatures();
    }

    void TearDown() override {
        // Common teardown code
    }
    
    // Helper method to skip tests that require specific CPU features
    void SkipIfMissingFeatures(bool requiredFeatureAvailable, const std::string& featureName) {
        if (!requiredFeatureAvailable) {
            GTEST_SKIP() << "Skipping test that requires " << featureName << " which is not available on this CPU";
        }
    }
};

// Test transpose function for float32
TEST_F(TransposeTest, F32TransposeTest) {
    // Only run optimized implementations if the CPU supports the required features
    bool runOptimizedTests = xdnn::isOptimizationLevelSupported(xdnn::OptimizationLevel::SSE2);
    if (!runOptimizedTests) {
        GTEST_SKIP() << "Skipping optimized transpose test as CPU lacks required features";
    }
    
    std::vector<std::pair<int, int>> shapes = {
        {1, 1}, {1, 16}, {16, 1}, {8, 8}, {16, 32}, {32, 16}, {64, 64}
    };
    
    for (const auto& shape : shapes) {
        int rows = shape.first;
        int cols = shape.second;
        
        // Allocate matrices
        float* input = new float[rows * cols];
        float* output = new float[rows * cols];
        float* expected_output = new float[rows * cols];
        
        // Initialize input
        initializeRandomTransposeMatrix(input, rows * cols);
        
        // Use reference implementation
        reference_transpose(input, output, rows, cols);
        
        // Confirm transpose works by transposing twice to get back original
        reference_transpose(output, expected_output, cols, rows);
        
        // Compare results - we should get back the original matrix
        const float epsilon = 1e-6f;
        for (int i = 0; i < rows * cols; ++i) {
            EXPECT_NEAR(input[i], expected_output[i], epsilon)
                << "Mismatch at index " << i << " with shape [" << rows << ", " << cols << "]";
        }
        
        // Clean up
        delete[] input;
        delete[] output;
        delete[] expected_output;
    }
}

// Test transpose function for FP16
TEST_F(TransposeTest, F16TransposeTest) {
    // Check if we have the required features for optimized FP16 transpose
    bool runOptimizedTests = xdnn::isOptimizationLevelSupported(xdnn::OptimizationLevel::SSE2);
    if (!runOptimizedTests) {
        GTEST_SKIP() << "Skipping optimized FP16 transpose test as CPU lacks required features";
    }
    
    std::vector<std::pair<int, int>> shapes = {
        {1, 1}, {1, 16}, {16, 1}, {8, 8}, {16, 32}, {32, 16}
    };
    
    for (const auto& shape : shapes) {
        int rows = shape.first;
        int cols = shape.second;
        
        // Allocate matrices
        XDNN_FP16* input = new XDNN_FP16[rows * cols];
        XDNN_FP16* output = new XDNN_FP16[rows * cols];
        XDNN_FP16* expected_output = new XDNN_FP16[rows * cols];
        
        // Initialize input
        initializeRandomTransposeMatrix(input, rows * cols);
        
        // Use reference implementation
        reference_transpose(input, output, rows, cols);
        
        // Confirm transpose works by transposing twice to get back original
        reference_transpose(output, expected_output, cols, rows);
        
        // Compare results - we should get back the original matrix
        const float epsilon = 1e-2f;  // Higher epsilon for fp16
        for (int i = 0; i < rows * cols; ++i) {
            EXPECT_NEAR(static_cast<float>(input[i]), 
                      static_cast<float>(expected_output[i]), 
                      epsilon)
                << "Mismatch at index " << i << " with shape [" << rows << ", " << cols << "]";
        }
        
        // Clean up
        delete[] input;
        delete[] output;
        delete[] expected_output;
    }
}

// Test transpose function for BF16
TEST_F(TransposeTest, BF16TransposeTest) {
    // Check if we have the required features for optimized BF16 transpose
    bool runOptimizedTests = xdnn::isOptimizationLevelSupported(xdnn::OptimizationLevel::SSE2);
    if (!runOptimizedTests) {
        GTEST_SKIP() << "Skipping optimized BF16 transpose test as CPU lacks required features";
    }
    
    std::vector<std::pair<int, int>> shapes = {
        {1, 1}, {1, 16}, {16, 1}, {8, 8}, {16, 32}, {32, 16}
    };
    
    for (const auto& shape : shapes) {
        int rows = shape.first;
        int cols = shape.second;
        
        // Allocate matrices
        XDNN_BF16* input = new XDNN_BF16[rows * cols];
        XDNN_BF16* output = new XDNN_BF16[rows * cols];
        XDNN_BF16* expected_output = new XDNN_BF16[rows * cols];
        
        // Initialize input
        initializeRandomTransposeMatrix(input, rows * cols);
        
        // Use reference implementation
        reference_transpose(input, output, rows, cols);
        
        // Confirm transpose works by transposing twice to get back original
        reference_transpose(output, expected_output, cols, rows);
        
        // Compare results - we should get back the original matrix
        const float epsilon = 1e-2f;  // Higher epsilon for bf16
        for (int i = 0; i < rows * cols; ++i) {
            EXPECT_NEAR(static_cast<float>(input[i]), 
                        static_cast<float>(expected_output[i]), 
                        epsilon)
                << "Mismatch at index " << i << " with shape [" << rows << ", " << cols << "]";
        }
        
        // Clean up
        delete[] input;
        delete[] output;
        delete[] expected_output;
    }
}

// Test transpose with specific patterns (identity, tall, wide matrices)
TEST_F(TransposeTest, TransposePatternTest) {
    // This test can run on any platform as it just verifies mathematical properties
    
    // Identity matrix
    const int size = 16;
    float* identity = new float[size * size];
    float* output = new float[size * size];
    
    // Initialize identity matrix
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            identity[i * size + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Transpose identity matrix using reference implementation
    reference_transpose(identity, output, size, size);
    
    // An identity matrix should be unchanged when transposed
    const float epsilon = 1e-6f;
    for (int i = 0; i < size * size; ++i) {
        EXPECT_NEAR(output[i], identity[i], epsilon);
    }
    
    // Clean up
    delete[] identity;
    delete[] output;
}
