#include "sgemm_f32f16bf16.h"
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

void fill_fp16_matrix(std::vector<XDNN_FP16>& mat, int rows, int cols, int ld, float start = 1.0f) {
    mat.resize(ld * rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            mat[r * ld + c] = static_cast<XDNN_FP16>(start + r * cols + c);
}

// Reference implementation for small_sgemm_f32f16bf16
// Note: This is a simple reference implementation for validation purposes
// The actual implementation has these constraints:
// - M must be 1
// - transB must be false
// - alpha must be 1.0
// - beta must be 0.0 or 1.0
void reference_small_sgemm_f32f16bf16(bool transB, int M, int N, int K, float alpha, 
                                      const float* A, int lda, const XDNN_FP16* B, int ldb, 
                                      float beta, XDNN_BF16* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = A[m * lda + k];
                float b_val = static_cast<float>(transB ? B[k + n * ldb] : B[k * ldb + n]);
                sum += a_val * b_val;
            }
            float c_val = (beta == 0.0f) ? 0.0f : beta * static_cast<float>(C[m * ldc + n]);
            C[m * ldc + n] = static_cast<XDNN_BF16>(alpha * sum + c_val);
        }
    }
}

// Tests for small_sgemm_f32f16bf16 function
// Note: According to the header comment and implementation constraints:
// - Must use transB=false
// - Must use M=1
// - alpha must be 1.0
// - beta must be 0.0 or 1.0
class SmallSgemmF32F16BF16Test : public ::testing::Test {
protected:
    // Test parameters
    bool transB = false;
    int M = 1; // As specified in the header comment, M should be 1
    int N = 16;
    int K = 64;
    float alpha = 1.0f;
    float beta = 0.0f;
    int lda = 64; // Leading dimension of A
    int ldb = 16; // Leading dimension of B
    int ldc = 16; // Leading dimension of C
    
    std::vector<float> A;
    std::vector<XDNN_FP16> B;
    std::vector<XDNN_BF16> C, C_ref;
    
    void SetUp() override {
        // Initialize matrices with test data
        fill_matrix(A, M, K, lda, 0.1f);
        fill_fp16_matrix(B, transB ? N : K, transB ? K : N, ldb, 0.2f);
        C.assign(M * ldc, static_cast<XDNN_BF16>(0.0f));
        C_ref.assign(M * ldc, static_cast<XDNN_BF16>(0.0f));
    }
};

// Test case for basic functionality with no transpose
TEST_F(SmallSgemmF32F16BF16Test, NoTranspose) {
    transB = false;
    reference_small_sgemm_f32f16bf16(transB, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C_ref.data(), ldc);
    small_sgemm_f32f16bf16(transB, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    
    for (int i = 0; i < M * N; ++i) {
        float c_val = bf16_to_f32(C[i]);
        float c_ref_val = bf16_to_f32(C_ref[i]);
        EXPECT_NEAR(c_val, c_ref_val, 1e-2) 
            << "Difference at index " << i << ": actual=" << c_val << ", expected=" << c_ref_val;
    }
}

// Test with beta=1
TEST_F(SmallSgemmF32F16BF16Test, WithBetaOne) {
    transB = false;
    beta = 1.0f; // library only supports beta = 0 or 1
    
    // Initialize C with non-zero values
    for (int i = 0; i < M * ldc; ++i) {
        C[i] = static_cast<XDNN_BF16>(0.5f);
        C_ref[i] = static_cast<XDNN_BF16>(0.5f);
    }
    
    reference_small_sgemm_f32f16bf16(transB, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C_ref.data(), ldc);
    small_sgemm_f32f16bf16(transB, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    
    for (int i = 0; i < M * N; ++i) {
        float c_val = bf16_to_f32(C[i]);
        float c_ref_val = bf16_to_f32(C_ref[i]);
        EXPECT_NEAR(c_val, c_ref_val, 1e-2)
            << "Difference at index " << i << " with beta=" << beta;
    }
}

// Test with different matrix dimensions
TEST_F(SmallSgemmF32F16BF16Test, DifferentDimensions) {
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
    fill_fp16_matrix(B, K, N, ldb, 0.2f);
    C.assign(M * N, static_cast<XDNN_BF16>(0.0f));
    C_ref.assign(M * N, static_cast<XDNN_BF16>(0.0f));
    
    reference_small_sgemm_f32f16bf16(transB, M, N, K, alpha, A.data(), K, B.data(), ldb, beta, C_ref.data(), N);
    small_sgemm_f32f16bf16(transB, M, N, K, alpha, A.data(), K, B.data(), ldb, beta, C.data(), N);
    
    for (int i = 0; i < M * N; ++i) {
        float c_val = bf16_to_f32(C[i]);
        float c_ref_val = bf16_to_f32(C_ref[i]);
        EXPECT_NEAR(c_val, c_ref_val, 1e-2)
            << "Difference at index " << i << " with N=" << N << ", K=" << K;
    }
}

// Test with matrix leading dimensions different from actual dimensions
TEST_F(SmallSgemmF32F16BF16Test, DifferentLeadingDimensions) {
    transB = false;
    // alpha must be 1 as required by the function
    alpha = 1.0f;
    
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
    fill_fp16_matrix(B, K, N, ldb, 0.2f);
    C.assign(M * ldc, static_cast<XDNN_BF16>(0.0f));
    C_ref.assign(M * ldc, static_cast<XDNN_BF16>(0.0f));
    
    reference_small_sgemm_f32f16bf16(transB, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C_ref.data(), ldc);
    small_sgemm_f32f16bf16(transB, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
    
    for (int i = 0; i < M * N; ++i) {
        float c_val = bf16_to_f32(C[i]);
        float c_ref_val = bf16_to_f32(C_ref[i]);
        EXPECT_NEAR(c_val, c_ref_val, 1e-2)
            << "Difference at index " << i << " with different leading dimensions";
    }
}

// Simple test with minimal dimensions (1x1 matrices)
TEST(SGEMM_F32F16BF16, SmallSGEMMMinimal) {
    int M = 1, N = 1, K = 1;
    float a = 2.0f;
    XDNN_FP16 b = static_cast<XDNN_FP16>(3.0f);
    XDNN_BF16 c = static_cast<XDNN_BF16>(0.0f);
    XDNN_BF16 c_ref = static_cast<XDNN_BF16>(0.0f);
    
    // Calculate reference result using the reference implementation
    reference_small_sgemm_f32f16bf16(false, M, N, K, 1.0f, &a, K, &b, N, 0.0f, &c_ref, N);
    
    // Call the function being tested
    small_sgemm_f32f16bf16(false, M, N, K, 1.0f, &a, K, &b, N, 0.0f, &c, N);
    
    // Compare results
    float c_val = bf16_to_f32(c);
    float c_ref_val = bf16_to_f32(c_ref);
    EXPECT_NEAR(c_val, c_ref_val, 1e-2);
}

// Test with specific values
TEST(SGEMM_F32F16BF16, SpecificValues) {
    int M = 1, N = 4, K = 4;
    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<XDNN_FP16> B(K * N);
    std::vector<XDNN_BF16> C(M * N, static_cast<XDNN_BF16>(0.0f));
    std::vector<XDNN_BF16> C_ref(M * N, static_cast<XDNN_BF16>(0.0f));
    
    // Initialize B with specific values
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            B[k * N + n] = static_cast<XDNN_FP16>(0.5f * (k + 1) * (n + 1));
        }
    }
    
    // Calculate reference result using the reference implementation
    reference_small_sgemm_f32f16bf16(false, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C_ref.data(), N);
    
    // Call the function being tested
    small_sgemm_f32f16bf16(false, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);
    
    // Compare results
    for (int n = 0; n < N; ++n) {
        float c_val = bf16_to_f32(C[n]);
        float c_ref_val = bf16_to_f32(C_ref[n]);
        EXPECT_NEAR(c_val, c_ref_val, 1e-2)
            << "Difference at index " << n;
    }
}