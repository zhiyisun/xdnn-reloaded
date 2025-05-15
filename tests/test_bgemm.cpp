#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include "bgemm_f32bf16f32.h"
#include "bgemm_bf16bf16bf16.h"
#include "data_types/bfloat16.h"
#include "../platform_detection.h"

// Helper function to initialize random matrices
template<typename T>
void initializeRandomBFMatrix(T* matrix, size_t size, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    for (size_t i = 0; i < size; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            matrix[i] = dist(gen);
        } else if constexpr (std::is_same_v<T, XDNN_BF16>) {
            matrix[i] = XDNN_BF16(dist(gen));
        } else {
            // Default case
            matrix[i] = static_cast<T>(dist(gen));
        }
    }
}

// Helper function for bfloat16 reference matrix multiplication
template<typename TA, typename TB, typename TC>
void reference_bgemm_mixed(bool transA, bool transB, int M, int N, int K,
                         float alpha, const TA* A, int lda, const TB* B, int ldb,
                         float beta, TC* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? 
                    static_cast<float>(A[k * lda + m]) : 
                    static_cast<float>(A[m * lda + k]);
                    
                float b_val = transB ? 
                    static_cast<float>(B[n * ldb + k]) : 
                    static_cast<float>(B[k * ldb + n]);
                    
                sum += a_val * b_val;
            }
            
            float c_val = static_cast<float>(C[m * ldc + n]);
            C[m * ldc + n] = static_cast<TC>(alpha * sum + beta * c_val);
        }
    }
}

// Test fixture for BGEMM tests
class BGEMMTest : public ::testing::Test {
protected:
    xdnn::CPUFeatures cpuFeatures;
    
    void SetUp() override {
        // Detect CPU features for conditional tests
        cpuFeatures = xdnn::detectCPUFeatures();
    }

    void TearDown() override {
        // Common teardown code
    }
    
    // Skip if not running on Intel Xeon
    void SkipIfNotXeon() {
        bool isXeon = cpuFeatures.supportsAVX512();
        if (!isXeon) {
            GTEST_SKIP() << "Skipping test that requires Intel Xeon platform which is not available";
        }
    }
};

// Test xdnn_bgemm_f32bf16f32 function
TEST_F(BGEMMTest, F32BF16F32Test) {
    // Skip if not running on Xeon platform
    SkipIfNotXeon();
    
    // Only continue with test setup if we haven't skipped
    if(::testing::Test::IsSkipped()) {
        return;
    }
    
    const int M = 32;
    const int N = 32;
    const int K = 32;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const bool transA = false;
    const bool transB = false;
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Allocate matrices
    float* A = new float[M * K];
    XDNN_BF16* B = new XDNN_BF16[K * N];
    float* C = new float[M * N];
    float* C_reference = new float[M * N];

    // Initialize matrices
    initializeRandomBFMatrix(A, M * K);
    initializeRandomBFMatrix(B, K * N);
    memset(C, 0, M * N * sizeof(float));
    memset(C_reference, 0, M * N * sizeof(float));

    // Call the function to test
    xdnn_bgemm_f32bf16f32(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    // Compute reference result
    reference_bgemm_mixed(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    // Compare results - using higher epsilon due to bf16 precision
    const float epsilon = 1e-2f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_reference[i], epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
}

// Test xdnn_bgemm_bf16bf16bf16 function
TEST_F(BGEMMTest, BF16BF16BF16Test) {
    // Skip if not running on Xeon platform
    SkipIfNotXeon();
    
    // Only continue with test setup if we haven't skipped
    if(::testing::Test::IsSkipped()) {
        return;
    }
    
    const int M = 32;
    const int N = 32;
    const int K = 32;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const bool transA = false;
    const bool transB = false;
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Allocate matrices
    XDNN_BF16* A = new XDNN_BF16[M * K];
    XDNN_BF16* B = new XDNN_BF16[K * N];
    XDNN_BF16* C = new XDNN_BF16[M * N];
    XDNN_BF16* C_reference = new XDNN_BF16[M * N];

    // Initialize matrices
    initializeRandomBFMatrix(A, M * K);
    initializeRandomBFMatrix(B, K * N);
    memset(C, 0, M * N * sizeof(XDNN_BF16));
    memset(C_reference, 0, M * N * sizeof(XDNN_BF16));

    // Call the function to test
    xdnn_bgemm_bf16bf16bf16(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    // Compute reference result
    reference_bgemm_mixed(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    // Compare results - using higher epsilon due to bf16 precision
    const float epsilon = 1e-1f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(static_cast<float>(C[i]), static_cast<float>(C_reference[i]), epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
}

// Test with different shapes, transposition flags, and alpha/beta values
TEST_F(BGEMMTest, TransposedBF16Test) {
    // Skip if not running on Xeon platform
    SkipIfNotXeon();
    
    // Only continue with test setup if we haven't skipped
    if(::testing::Test::IsSkipped()) {
        return;
    }
    
    const int M = 20;
    const int N = 24;
    const int K = 28;
    const float alpha = 1.5f;
    const float beta = 0.5f;
    const bool transA = true;
    const bool transB = true;
    const int lda = M;  // Transposed dimensions
    const int ldb = K;  // Transposed dimensions
    const int ldc = N;

    // Allocate matrices
    XDNN_BF16* A = new XDNN_BF16[K * M];  // K x M because transposed
    XDNN_BF16* B = new XDNN_BF16[N * K];  // N x K because transposed
    XDNN_BF16* C = new XDNN_BF16[M * N];
    XDNN_BF16* C_reference = new XDNN_BF16[M * N];

    // Initialize matrices
    initializeRandomBFMatrix(A, K * M);
    initializeRandomBFMatrix(B, N * K);
    initializeRandomBFMatrix(C, M * N);  // Non-zero initial C for testing beta
    
    // Copy C for reference
    for (int i = 0; i < M * N; ++i) {
        C_reference[i] = C[i];
    }

    // Call the function to test
    xdnn_bgemm_bf16bf16bf16(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    // Compute reference result
    reference_bgemm_mixed(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    // Compare results - using higher epsilon due to bf16 precision
    const float epsilon = 1e-1f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(static_cast<float>(C[i]), static_cast<float>(C_reference[i]), epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
}
