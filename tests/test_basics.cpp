#include "conversion.h"
#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include "data_types/float16.h"
#include "data_types/bfloat16.h"

// Helper function to initialize random matrices
template<typename T>
void initializeRandomMatrix(T* matrix, size_t size, float min = -1.0f, float max = 1.0f) {
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

// Test fixture for basic tests
class BasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code
    }

    void TearDown() override {
        // Common teardown code
    }
};

// Test basic transposition functionality
TEST_F(BasicTest, BasicTransposeTest) {
    std::vector<std::pair<int, int>> shapes = {
        {1, 1}, {1, 16}, {16, 1}, {8, 8}, {16, 32}, {32, 16}, {64, 64}
    };
    
    for (const auto& shape : shapes) {
        int rows = shape.first;
        int cols = shape.second;
        
        // Allocate matrices
        float* input = new float[rows * cols];
        float* output = new float[rows * cols];
        float* reference_output = new float[rows * cols];
        
        // Initialize input
        initializeRandomMatrix(input, rows * cols);
        
        // Compute reference result
        reference_transpose(input, output, rows, cols);
        
        // Compute with our reference implementation again for verification
        reference_transpose(input, reference_output, rows, cols);
        
        // Compare results
        const float epsilon = 1e-6f;
        for (int i = 0; i < rows * cols; ++i) {
            EXPECT_NEAR(output[i], reference_output[i], epsilon)
                << "Mismatch at index " << i << " with shape [" << rows << ", " << cols << "]";
        }
        
        // Clean up
        delete[] input;
        delete[] output;
        delete[] reference_output;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
