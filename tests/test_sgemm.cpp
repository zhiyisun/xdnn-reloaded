#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include "sgemm.h"

// Helper function to initialize random matrices
void initializeRandomMatrix(float* matrix, size_t size, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    for (size_t i = 0; i < size; ++i) {
        matrix[i] = dist(gen);
    }
}

// Helper function for simple matrix multiplication (for validation)
void reference_sgemm(bool transA, bool transB, int M, int N, int K,
                     float alpha, const float* A, int lda, const float* B, int ldb,
                     float beta, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                float b_val = transB ? B[n * ldb + k] : B[k * ldb + n];
                sum += a_val * b_val;
            }
            C[m * ldc + n] = alpha * sum + beta * C[m * ldc + n];
        }
    }
}

#include "../platform_detection.h"

// Test fixture for SGEMM tests
class SGEMMTest : public ::testing::Test {
protected:
    xdnn::CPUFeatures cpuFeatures;
    
    void SetUp() override {
        // Detect CPU features for conditional tests
        cpuFeatures = xdnn::detectCPUFeatures();
    }

    void TearDown() override {
        // Common teardown code
    }
    
    // Helper method to skip tests that require Intel Xeon
    void SkipIfNotXeon() {
        bool isXeon = cpuFeatures.supportsAVX512();
        if (!isXeon) {
            GTEST_SKIP() << "Skipping test that requires Intel Xeon platform which is not available";
        }
    }
};

// Test xdnn_sgemm function
TEST_F(SGEMMTest, BasicSGEMMTest) {
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
    float* B = new float[K * N];
    float* C = new float[M * N];
    float* C_reference = new float[M * N];

    // Initialize matrices
    initializeRandomMatrix(A, M * K);
    initializeRandomMatrix(B, K * N);
    memset(C, 0, M * N * sizeof(float));
    memset(C_reference, 0, M * N * sizeof(float));

    // Call the function to test
    xdnn_sgemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    // Compute reference result
    reference_sgemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    // Compare results
    const float epsilon = 1e-4f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_reference[i], epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
}

// Test xdnn_sgemm_single_thread function
TEST_F(SGEMMTest, SingleThreadSGEMMTest) {
    // Skip if not running on Xeon platform
    SkipIfNotXeon();
    
    // Only continue with test setup if we haven't skipped
    if(::testing::Test::IsSkipped()) {
        return;
    }
    
    const int M = 16;
    const int N = 16;
    const int K = 16;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const bool transA = false;
    const bool transB = false;
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Allocate matrices
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    float* C_reference = new float[M * N];

    // Initialize matrices
    initializeRandomMatrix(A, M * K);
    initializeRandomMatrix(B, K * N);
    memset(C, 0, M * N * sizeof(float));
    memset(C_reference, 0, M * N * sizeof(float));

    // Call the function to test
    xdnn_sgemm_single_thread(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    // Compute reference result
    reference_sgemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    // Compare results
    const float epsilon = 1e-4f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_reference[i], epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
}

// Test xdnn_sgemm_packb and xdnn_sgemm_compute functions
TEST_F(SGEMMTest, PackBAndComputeTest) {
    // Skip if not running on Xeon platform
    SkipIfNotXeon();
    
    // Only continue with test setup if we haven't skipped
    if(::testing::Test::IsSkipped()) {
        return;
    }
    
    const int M = 24;
    const int N = 24;
    const int K = 24;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const bool transA = false;
    const bool transB = false;
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Allocate matrices
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    float* C_reference = new float[M * N];
    float* packedB = new float[K * N + 16]; // Extra space for alignment

    // Initialize matrices
    initializeRandomMatrix(A, M * K);
    initializeRandomMatrix(B, K * N);
    memset(C, 0, M * N * sizeof(float));
    memset(C_reference, 0, M * N * sizeof(float));

    // Pack matrix B
    xdnn_sgemm_packb(transB, N, K, B, ldb, packedB);
    
    // Call the compute function
    xdnn_sgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);

    // Compute reference result
    reference_sgemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    // Compare results
    const float epsilon = 1e-4f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_reference[i], epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
    delete[] packedB;
}

// Test xdnn_sgemm_compute_silu function
TEST_F(SGEMMTest, ComputeSiluTest) {
    // Skip if not running on Xeon platform
    SkipIfNotXeon();
    
    // Only continue with test setup if we haven't skipped
    if(::testing::Test::IsSkipped()) {
        return;
    }
    
    const int M = 16;
    const int N = 16;
    const int K = 16;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const bool transA = false;
    const bool transB = false;
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Allocate matrices
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    float* C_reference = new float[M * N];
    float* packedB = new float[K * N + 16]; // Extra space for alignment

    // Initialize matrices
    initializeRandomMatrix(A, M * K);
    initializeRandomMatrix(B, K * N);
    memset(C, 0, M * N * sizeof(float));
    memset(C_reference, 0, M * N * sizeof(float));

    // Pack matrix B
    xdnn_sgemm_packb(transB, N, K, B, ldb, packedB);
    
    // Call the silu compute function
    xdnn_sgemm_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);

    // Compute reference result (matmul followed by SiLU activation)
    reference_sgemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);
    
    // Apply SiLU: x * sigmoid(x)
    for (int i = 0; i < M * N; ++i) {
        float sigmoid_val = 1.0f / (1.0f + exp(-C_reference[i]));
        C_reference[i] = C_reference[i] * sigmoid_val;
    }

    // Compare results
    const float epsilon = 1e-4f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_reference[i], epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
    delete[] packedB;
}

// Additional tests with different shapes, transposition flags, and alpha/beta values
TEST_F(SGEMMTest, TransposedMatricesTest) {
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
    const int lda = M;  // transposed dimensions
    const int ldb = K;  // transposed dimensions
    const int ldc = N;

    // Allocate matrices
    float* A = new float[K * M];  // K x M because transposed
    float* B = new float[N * K];  // N x K because transposed
    float* C = new float[M * N];
    float* C_reference = new float[M * N];

    // Initialize matrices
    initializeRandomMatrix(A, K * M);
    initializeRandomMatrix(B, N * K);
    initializeRandomMatrix(C, M * N);  // Non-zero initial C for testing beta
    
    // Copy C for reference
    memcpy(C_reference, C, M * N * sizeof(float));

    // Call the function to test
    xdnn_sgemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    // Compute reference result
    reference_sgemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    // Compare results
    const float epsilon = 1e-4f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_reference[i], epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
}
