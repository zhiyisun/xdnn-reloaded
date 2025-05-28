#include "sgemm_bf16bf16f32.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

// Reference: simple BF16 emulation for testing verification
float bf16_to_f32(XDNN_BF16 bf16_val) {
    return static_cast<float>(bf16_val);
}

// Helper to fill matrix with a pattern
void fill_matrix(std::vector<float>& mat, int rows, int cols, int ld, float start = 1.0f) {
    mat.resize(ld * rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            mat[r * ld + c] = start + r * cols + c;
}

void fill_bf16_matrix(std::vector<XDNN_BF16>& mat, int rows, int cols, int ld, float start = 1.0f) {
    mat.resize(ld * rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            mat[r * ld + c] = static_cast<XDNN_BF16>(start + r * cols + c);
}

// Reference implementation for small_sgemm_bf16bf16f32_b (paged attention version)
void reference_small_sgemm_bf16bf16f32_b(bool transB, int M, int N, int K, const XDNN_BF16* A, int lda,
                                        const XDNN_BF16* B, int ldb, float* C, int ldc,
                                        int* blockIndices, int blockStride, int blockSize) {
    // This reference implementation only works for M=1 to match the library constraints
    if (M != 1) return;

    // Reset output to zeros
    for (int n = 0; n < N; n++) {
        C[n] = 0.0f;
    }

    // The paged attention model works by dividing the N dimension into blocks
    // Each block is blockSize columns wide
    int numBlocks = (N + blockSize - 1) / blockSize; // Ceiling division
    for (int b = 0; b < numBlocks; b++) {
        // Get the block index from the provided indices array
        int blockIdx = blockIndices[b];
        
        // Determine start position in the output 
        int outputOffset = b * blockSize;
        
        // Determine start position in B based on block index
        const XDNN_BF16* blockB = B + blockIdx * blockStride;
        
        // Process each column in this block (but not exceeding N)
        int columnsInThisBlock = std::min(blockSize, N - outputOffset);
        for (int j = 0; j < columnsInThisBlock; j++) {
            float sum = 0.0f;
            
            // Do the matrix multiplication for this column
            for (int k = 0; k < K; k++) {
                float a_val = bf16_to_f32(A[k]);
                
                // Get B value - note the access pattern depends on transB
                float b_val;
                if (transB) {
                    b_val = bf16_to_f32(blockB[j * ldb + k]);
                } else {
                    b_val = bf16_to_f32(blockB[k * ldb + j]);
                }
                
                sum += a_val * b_val;
            }
            
            // Store the result in the output
            C[outputOffset + j] = sum;
        }
    }
}

// Tests for small_sgemm_bf16bf16f32_b function (paged attention)
class SmallSgemmBF16BF16F32BlockTest : public ::testing::Test {
protected:
    // Test parameters
    bool transB = true;  // As specified in the header comment, transB should be true
    int M = 1;           // As specified in the header comment, M should be 1
    int N = 16;
    int K = 16;          // Use a smaller K for testing
    int lda = 16;        // Leading dimension of A
    int ldb = 8;         // Leading dimension of B
    int ldc = 16;        // Leading dimension of C
    int blockSize = 4;   // Block size as shown in the header diagram
    int numBlocks = 0;   // Will be calculated in setup
    int blockStride = 32; // Stride between blocks
    
    std::vector<XDNN_BF16> A;
    std::vector<XDNN_BF16> B;
    std::vector<float> C, C_ref;
    std::vector<int> blockIndices;
    
    void SetUp() override {
        // Calculate number of blocks based on ceiling division
        numBlocks = (N + blockSize - 1) / blockSize;
        
        // Initialize matrices with test data
        A.resize(K);
        for (int k = 0; k < K; ++k) {
            A[k] = static_cast<XDNN_BF16>(0.1f * (k+1));
        }
        
        // Calculate the total size needed for B - ensure we allocate enough memory
        // Each block needs K * ldb elements
        size_t b_size = numBlocks * (blockStride + blockSize);
        B.resize(b_size, static_cast<XDNN_BF16>(0.0f));
        
        // Clear B first to ensure clean state
        for (size_t i = 0; i < B.size(); ++i) {
            B[i] = static_cast<XDNN_BF16>(0.0f);
        }
        
        // Use a constant value for each block to make debugging easier
        float blockValue = 0.5f;
        for (int b = 0; b < numBlocks; ++b) {
            for (int n = 0; n < blockSize && (b * blockSize + n) < N; ++n) {
                for (int k = 0; k < K; ++k) {
                    // For transB=true, the layout changes
                    int idx = b * blockStride + n * ldb + k;
                    if (idx < static_cast<int>(B.size())) {
                        B[idx] = static_cast<XDNN_BF16>(blockValue);
                    }
                }
            }
        }
        
        C.assign(N, 0.0f);
        C_ref.assign(N, 0.0f);
        
        // Initialize block indices (default sequential blocks)
        blockIndices.resize(numBlocks);
        for (int i = 0; i < numBlocks; ++i) {
            blockIndices[i] = i;
        }
    }
};

// Test case for basic paged attention functionality
TEST_F(SmallSgemmBF16BF16F32BlockTest, BasicPagedAttention) {
    // Call the reference implementation
    reference_small_sgemm_bf16bf16f32_b(transB, M, N, K, A.data(), lda, 
                                       B.data(), ldb, C_ref.data(), ldc,
                                       blockIndices.data(), blockStride, blockSize);
    
    // Call the function being tested
    small_sgemm_bf16bf16f32_b(transB, M, N, K, A.data(), lda, 
                             B.data(), ldb, C.data(), ldc,
                             blockIndices.data(), blockStride, blockSize);
    
    // Compare results
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-2) 
            << "Difference at index " << i << ": actual=" << C[i] << ", expected=" << C_ref[i];
    }
}

// Test with non-sequential block indices
TEST_F(SmallSgemmBF16BF16F32BlockTest, NonSequentialBlocks) {
    // Modify block indices to be non-sequential
    if (numBlocks >= 2) {
        // Swap first two block indices if we have at least 2 blocks
        std::swap(blockIndices[0], blockIndices[1]);
    }
    
    // Call the reference implementation
    reference_small_sgemm_bf16bf16f32_b(transB, M, N, K, A.data(), lda, 
                                       B.data(), ldb, C_ref.data(), ldc,
                                       blockIndices.data(), blockStride, blockSize);
    
    // Call the function being tested
    small_sgemm_bf16bf16f32_b(transB, M, N, K, A.data(), lda, 
                             B.data(), ldb, C.data(), ldc,
                             blockIndices.data(), blockStride, blockSize);
    
    // Compare results
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-2) 
            << "Difference at index " << i << " with non-sequential blocks";
    }
}

// Test with different block size and stride
TEST_F(SmallSgemmBF16BF16F32BlockTest, DifferentBlockSizeStride) {
    // Modify block size and stride
    blockSize = 8;
    blockStride = 64;
    
    // Recalculate number of blocks
    numBlocks = (N + blockSize - 1) / blockSize;
    blockIndices.resize(numBlocks);
    for (int i = 0; i < numBlocks; ++i) {
        blockIndices[i] = i;
    }
    
    // Reinitialize B with new block layout
    size_t b_size = numBlocks * (blockStride + blockSize);
    B.resize(b_size, static_cast<XDNN_BF16>(0.0f));
    
    float blockValue = 0.3f;
    for (int b = 0; b < numBlocks; ++b) {
        for (int n = 0; n < blockSize && (b * blockSize + n) < N; ++n) {
            for (int k = 0; k < K; ++k) {
                // For transB=true, the layout changes
                int idx = b * blockStride + n * ldb + k;
                if (idx < static_cast<int>(B.size())) {
                    B[idx] = static_cast<XDNN_BF16>(blockValue);
                }
            }
        }
    }
    
    // Reset outputs
    C.assign(N, 0.0f);
    C_ref.assign(N, 0.0f);
    
    // Call the reference implementation
    reference_small_sgemm_bf16bf16f32_b(transB, M, N, K, A.data(), lda, 
                                       B.data(), ldb, C_ref.data(), ldc,
                                       blockIndices.data(), blockStride, blockSize);
    
    // Call the function being tested
    small_sgemm_bf16bf16f32_b(transB, M, N, K, A.data(), lda, 
                             B.data(), ldb, C.data(), ldc,
                             blockIndices.data(), blockStride, blockSize);
    
    // Compare results
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-2) 
            << "Difference at index " << i << " with different block size and stride";
    }
}

// Test with specific block values
TEST_F(SmallSgemmBF16BF16F32BlockTest, SpecificBlockValues) {
    // Smaller test dimensions for readability
    N = 8;
    K = 4;
    blockSize = 2;
    blockStride = 16;
    ldb = 4;
    ldc = 8;
    
    // Recalculate number of blocks
    numBlocks = (N + blockSize - 1) / blockSize;
    blockIndices.resize(numBlocks);
    for (int i = 0; i < numBlocks; ++i) {
        blockIndices[i] = i;
    }
    
    // Reinitialize matrices with specific values
    A.resize(K);
    for (int k = 0; k < K; ++k) {
        A[k] = static_cast<XDNN_BF16>(1.0f);  // All ones for simplicity
    }
    
    size_t b_size = numBlocks * blockStride;
    B.resize(b_size, static_cast<XDNN_BF16>(0.0f));
    
    // Set specific values in each block to make validation easier
    for (int b = 0; b < numBlocks; ++b) {
        for (int n = 0; n < blockSize && (b * blockSize + n) < N; ++n) {
            for (int k = 0; k < K; ++k) {
                // For transB=true, column-major layout within each block
                int idx = b * blockStride + n * ldb + k;
                if (idx < static_cast<int>(B.size())) {
                    // Use block index + 1 as the value
                    B[idx] = static_cast<XDNN_BF16>(static_cast<float>(b + 1));
                }
            }
        }
    }
    
    // Reset outputs
    C.assign(N, 0.0f);
    C_ref.assign(N, 0.0f);
    
    // Call the reference implementation
    reference_small_sgemm_bf16bf16f32_b(transB, M, N, K, A.data(), K, 
                                       B.data(), ldb, C_ref.data(), ldc,
                                       blockIndices.data(), blockStride, blockSize);
    
    // Call the function being tested
    small_sgemm_bf16bf16f32_b(transB, M, N, K, A.data(), K, 
                             B.data(), ldb, C.data(), ldc,
                             blockIndices.data(), blockStride, blockSize);
    
    // Compare results and also print them for validation
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-2) 
            << "Difference at index " << i << ": actual=" << C[i] << ", expected=" << C_ref[i];
    }
}

// Entry point for the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
