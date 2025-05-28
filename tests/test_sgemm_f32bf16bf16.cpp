#include "sgemm_f32bf16bf16.h"
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

// Reference implementation for small_sgemm_f32bf16bf16
// Note: This is a simple reference implementation for validation purposes
// The actual implementation has these constraints:
// - M must be 1
// - transB must be false
void reference_small_sgemm_f32bf16bf16(bool transB, int M, int N, int K, const float* A, int lda, 
                                      const XDNN_BF16* B, int ldb, XDNN_BF16* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = A[m * lda + k];
                float b_val = bf16_to_f32(transB ? B[k + n * ldb] : B[k * ldb + n]);
                sum += a_val * b_val;
            }
            C[m * ldc + n] = static_cast<XDNN_BF16>(sum);
        }
    }
}

// Reference implementation for small_sgemm_f32bf16bf16_b (paged attention version)
void reference_small_sgemm_f32bf16bf16_b(bool transB, int M, int N, int K, const float* A, int lda,
                                        const XDNN_BF16* B, int ldb, XDNN_BF16* C, int ldc,
                                        int* blockIndices, int blockStride, int blockSize) {
    // This reference implementation only works for M=1 to match the library constraints
    if (M != 1) return;

    // Reset output to zeros
    for (int n = 0; n < N; n++) {
        C[n] = static_cast<XDNN_BF16>(0.0f);
    }

    // The paged attention model works by dividing the N dimension into blocks
    // Each block is blockSize columns wide
    // The actual implementation of small_sgemm_f32bf16bf16_b appears to handle
    // one block at a time, which differs from our original assumption

    // For each complete block
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
                float a_val = A[k];
                
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
            C[outputOffset + j] = static_cast<XDNN_BF16>(sum);
        }
    }
}

// Tests for small_sgemm_f32bf16bf16 function
// Note: According to the header comment and implementation constraints:
// - Must use transB=false
// - Must use M=1
class SmallSgemmF32BF16BF16Test : public ::testing::Test {
protected:
    // Test parameters
    bool transB = false;
    int M = 1; // As specified in the header comment, M should be 1
    int N = 16;
    int K = 64;
    int lda = 64; // Leading dimension of A
    int ldb = 16; // Leading dimension of B
    int ldc = 16; // Leading dimension of C
    
    std::vector<float> A;
    std::vector<XDNN_BF16> B;
    std::vector<XDNN_BF16> C, C_ref;
    
    void SetUp() override {
        // Initialize matrices with test data
        fill_matrix(A, M, K, lda, 0.1f);
        fill_bf16_matrix(B, K, N, ldb, 0.2f);
        C.assign(M * ldc, static_cast<XDNN_BF16>(0.0f));
        C_ref.assign(M * ldc, static_cast<XDNN_BF16>(0.0f));
    }
};

// Test case for basic functionality with no transpose
TEST_F(SmallSgemmF32BF16BF16Test, NoTranspose) {
    transB = false;
    reference_small_sgemm_f32bf16bf16(transB, M, N, K, A.data(), lda, B.data(), ldb, C_ref.data(), ldc);
    small_sgemm_f32bf16bf16(transB, M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc);
    
    for (int i = 0; i < M * N; ++i) {
        float c_val = bf16_to_f32(C[i]);
        float c_ref_val = bf16_to_f32(C_ref[i]);
        EXPECT_NEAR(c_val, c_ref_val, 1e-2) 
            << "Difference at index " << i << ": actual=" << c_val << ", expected=" << c_ref_val;
    }
}

// Test with different matrix dimensions
TEST_F(SmallSgemmF32BF16BF16Test, DifferentDimensions) {
    // Keep transB=false as required by the function
    transB = false;
    
    // Use different N and K dimensions
    N = 32;
    K = 32;
    ldb = N;
    
    // Reinitialize matrices with new dimensions
    A.clear();
    B.clear();
    C.clear();
    C_ref.clear();
    
    fill_matrix(A, M, K, K, 0.1f);
    fill_bf16_matrix(B, K, N, ldb, 0.2f);
    C.assign(M * N, static_cast<XDNN_BF16>(0.0f));
    C_ref.assign(M * N, static_cast<XDNN_BF16>(0.0f));
    
    reference_small_sgemm_f32bf16bf16(transB, M, N, K, A.data(), K, B.data(), ldb, C_ref.data(), N);
    small_sgemm_f32bf16bf16(transB, M, N, K, A.data(), K, B.data(), ldb, C.data(), N);
    
    for (int i = 0; i < M * N; ++i) {
        float c_val = bf16_to_f32(C[i]);
        float c_ref_val = bf16_to_f32(C_ref[i]);
        EXPECT_NEAR(c_val, c_ref_val, 1e-2)
            << "Difference at index " << i << " with N=" << N << ", K=" << K;
    }
}

// Test with matrix leading dimensions different from actual dimensions
TEST_F(SmallSgemmF32BF16BF16Test, DifferentLeadingDimensions) {
    transB = false;
    
    // Set leading dimensions larger than actual dimensions
    lda = K + 8;
    ldb = N + 4;
    ldc = N + 4;
    
    // Reinitialize matrices with new leading dimensions
    A.clear();
    B.clear();
    C.clear();
    C_ref.clear();
    
    fill_matrix(A, M, K, lda, 0.1f);
    fill_bf16_matrix(B, K, N, ldb, 0.2f);
    C.assign(M * ldc, static_cast<XDNN_BF16>(0.0f));
    C_ref.assign(M * ldc, static_cast<XDNN_BF16>(0.0f));
    
    reference_small_sgemm_f32bf16bf16(transB, M, N, K, A.data(), lda, B.data(), ldb, C_ref.data(), ldc);
    small_sgemm_f32bf16bf16(transB, M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc);
    
    for (int i = 0; i < M * N; ++i) {
        float c_val = bf16_to_f32(C[i]);
        float c_ref_val = bf16_to_f32(C_ref[i]);
        EXPECT_NEAR(c_val, c_ref_val, 1e-2)
            << "Difference at index " << i << " with different leading dimensions";
    }
}

// Simple test with minimal dimensions (1x1 matrices)
TEST(SGEMM_F32BF16BF16, SmallSGEMMMinimal) {
    int M = 1, N = 1, K = 1;
    float a = 2.0f;
    XDNN_BF16 b = static_cast<XDNN_BF16>(3.0f);
    XDNN_BF16 c = static_cast<XDNN_BF16>(0.0f);
    XDNN_BF16 c_ref = static_cast<XDNN_BF16>(0.0f);
    
    // Calculate reference result using the reference implementation
    reference_small_sgemm_f32bf16bf16(false, M, N, K, &a, K, &b, N, &c_ref, N);
    
    // Call the function being tested
    small_sgemm_f32bf16bf16(false, M, N, K, &a, K, &b, N, &c, N);
    
    // Compare results
    float c_val = bf16_to_f32(c);
    float c_ref_val = bf16_to_f32(c_ref);
    EXPECT_NEAR(c_val, c_ref_val, 1e-2);
}

// Test with specific values
TEST(SGEMM_F32BF16BF16, SpecificValues) {
    int M = 1, N = 4, K = 4;
    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<XDNN_BF16> B(K * N);
    std::vector<XDNN_BF16> C(M * N, static_cast<XDNN_BF16>(0.0f));
    std::vector<XDNN_BF16> C_ref(M * N, static_cast<XDNN_BF16>(0.0f));
    
    // Initialize B with specific values
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            B[k * N + n] = static_cast<XDNN_BF16>(0.5f * (k + 1) * (n + 1));
        }
    }
    
    // Calculate reference result using the reference implementation
    reference_small_sgemm_f32bf16bf16(false, M, N, K, A.data(), K, B.data(), N, C_ref.data(), N);
    
    // Call the function being tested
    small_sgemm_f32bf16bf16(false, M, N, K, A.data(), K, B.data(), N, C.data(), N);
    
    // Compare results
    for (int n = 0; n < N; ++n) {
        float c_val = bf16_to_f32(C[n]);
        float c_ref_val = bf16_to_f32(C_ref[n]);
        EXPECT_NEAR(c_val, c_ref_val, 1e-2)
            << "Difference at index " << n;
    }
}

// Tests for small_sgemm_f32bf16bf16_b function (paged attention)
class SmallSgemmF32BF16BF16BlockTest : public ::testing::Test {
protected:
    // Test parameters
    bool transB = false;
    int M = 1; // As specified in the header comment, M should be 1
    int N = 16;
    int K = 16; // Use a smaller K for testing
    int lda = 16; // Leading dimension of A
    int ldb = 8; // Leading dimension of B
    int ldc = 16; // Leading dimension of C
    int blockSize = 4; // Block size as shown in the header diagram
    int numBlocks = 0; // Will be calculated in setup
    int blockStride = 32; // Smaller stride between blocks
    
    std::vector<float> A;
    std::vector<XDNN_BF16> B;
    std::vector<XDNN_BF16> C, C_ref;
    std::vector<int> blockIndices;
    
    void SetUp() override {
        // Calculate number of blocks based on ceiling division
        numBlocks = (N + blockSize - 1) / blockSize;
        
        // Initialize matrices with test data - use constant values to simplify debugging
        A.resize(K);
        for (int k = 0; k < K; ++k) {
            A[k] = 0.1f * (k+1);
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
            for (int k = 0; k < K; ++k) {
                for (int n = 0; n < blockSize && (b * blockSize + n) < N; ++n) {
                    int idx = k * ldb + b * blockStride + n;
                    if (idx < static_cast<int>(B.size())) {
                        B[idx] = static_cast<XDNN_BF16>(blockValue);
                    }
                }
            }
        }
        
        C.assign(N, static_cast<XDNN_BF16>(0.0f));
        C_ref.assign(N, static_cast<XDNN_BF16>(0.0f));
        
        // Initialize block indices (default sequential blocks)
        blockIndices.resize(numBlocks);
        for (int i = 0; i < numBlocks; ++i) {
            blockIndices[i] = i;
        }
    }
};
