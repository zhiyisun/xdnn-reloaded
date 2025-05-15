#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include "amx_sgemm_bf16bf16bf16.h"
#include "amx_sgemm_bf16f8bf16.h"
#include "data_types/bfloat16.h"
#include "data_types/fp8_e4m3.h" // Using XDNN_E4M3
#include "../platform_detection.h"

// Helper function to initialize random matrices
template<typename T>
void initializeRandomAMXMatrix(T* matrix, size_t size, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    for (size_t i = 0; i < size; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            matrix[i] = dist(gen);
        } else if constexpr (std::is_same_v<T, XDNN_BF16>) {
            matrix[i] = XDNN_BF16(dist(gen));
        } else if constexpr (std::is_same_v<T, XDNN_E4M3>) {
            matrix[i] = XDNN_E4M3(dist(gen));
        } else {
            // Default case
            matrix[i] = static_cast<T>(dist(gen));
        }
    }
}

// Helper function for AMX reference matrix multiplication
template<typename TA, typename TB, typename TC>
void reference_amx_gemm_mixed(bool transA, bool transB, int M, int N, int K,
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

// Test fixture for AMX SGEMM tests
class AMXSGEMMTest : public ::testing::Test {
protected:
    xdnn::CPUFeatures cpuFeatures;
    
    void SetUp() override {
        // Detect CPU features for conditional tests
        cpuFeatures = xdnn::detectCPUFeatures();
    }

    void TearDown() override {
        // Common teardown code
    }
    
    // Skip test if AMX isn't supported
    void SkipIfAMXNotSupported() {
        // AMX requires AVX-512 foundation and more
        bool isSupported = cpuFeatures.supportsAVX512();
        if (!isSupported) {
            GTEST_SKIP() << "AMX instructions not supported on this CPU";
        }
    }
    
    // Skip if not running on Intel Xeon
    void SkipIfNotXeon() {
        bool isXeon = cpuFeatures.supportsAVX512();
        if (!isXeon) {
            GTEST_SKIP() << "Skipping test that requires Intel Xeon platform which is not available";
        }
    }
};

// Test xdnn_amx_sgemm_bf16bf16bf16 function
TEST_F(AMXSGEMMTest, BF16BF16BF16Test) {
    // MUST skip before even defining variables because some AMX functions
    // may be called during initialization/compilation
    SkipIfAMXNotSupported();
    
    // Only continue with test setup if we haven't skipped
    // This prevents crashes when AMX is not available
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
    initializeRandomAMXMatrix(A, M * K);
    initializeRandomAMXMatrix(B, K * N);
    memset(C, 0, M * N * sizeof(XDNN_BF16));
    memset(C_reference, 0, M * N * sizeof(XDNN_BF16));

    // Call the function to test
    // Pack B matrix first
    int pack_size = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(N, K, 16, 16);
    XDNN_BF16* packedB = new XDNN_BF16[pack_size];
    xdnn_small_amx_sgemm_bf16bf16bf16_packb(transB, N, K, B, ldb, packedB, pack_size);
    
    // Then compute the result
    xdnn_small_amx_sgemm_bf16bf16bf16_compute(M, N, K, A, lda, packedB, K, C, ldc, beta);

    // Compute reference result
    reference_amx_gemm_mixed(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    // Compare results - using higher epsilon due to bf16 precision and potential AMX differences
    const float epsilon = 1e-1f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(static_cast<float>(C[i]), static_cast<float>(C_reference[i]), epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
    // packedB was already deleted after use
}

// Test xdnn_amx_sgemm_bf16f8bf16 function
TEST_F(AMXSGEMMTest, BF16F8BF16Test) {
    // MUST skip before even defining variables because some AMX functions
    // may be called during initialization/compilation
    SkipIfAMXNotSupported();
    
    // Only continue with test setup if we haven't skipped
    // This prevents crashes when AMX is not available
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
    XDNN_E4M3* B = new XDNN_E4M3[K * N];
    XDNN_BF16* C = new XDNN_BF16[M * N];
    XDNN_BF16* C_reference = new XDNN_BF16[M * N];
    float* scales = new float[N];  // Scales for fp8 quantization

    // Initialize matrices
    initializeRandomAMXMatrix(A, M * K);
    initializeRandomAMXMatrix(B, K * N);
    memset(C, 0, M * N * sizeof(XDNN_BF16));
    memset(C_reference, 0, M * N * sizeof(XDNN_BF16));
    
    // Initialize scales (normally these would be computed from quantization)
    for (int i = 0; i < N; i++) {
        scales[i] = 0.1f;  // Simple scale factor for testing
    }

    // Call the function to test
    // Pack B matrix first
    int pack_size = 64; // AMX uses 64 as pack size
    int packb_size = xdnn_small_amx_sgemm_bf16f8bf16_packb_size(N, K, pack_size);
    XDNN_E4M3* packedB = new XDNN_E4M3[packb_size];
    xdnn_small_amx_sgemm_bf16f8bf16_packb(transB, N, K, B, ldb, packedB, pack_size);
    
    // Then compute the result with additional required parameters
    int lds = 1; // Stride for scaling factors (assuming contiguous)
    int blockSize = 1; // For simplicity in testing
    float* bias = nullptr; // No bias for testing
    xdnn_small_amx_sgemm_bf16f8bf16_compute(M, N, K, A, lda, packedB, 
                                          C, ldc, scales, lds, blockSize, alpha, beta, bias);

    // Compute reference result with scaling
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? 
                    static_cast<float>(A[k * lda + m]) : 
                    static_cast<float>(A[m * lda + k]);
                    
                float raw_b_val = transB ? 
                    static_cast<float>(B[n * ldb + k]) : 
                    static_cast<float>(B[k * ldb + n]);
                
                // Apply scaling for fp8
                float b_val = raw_b_val * scales[n];
                    
                sum += a_val * b_val;
            }
            
            float c_val = static_cast<float>(C_reference[m * ldc + n]);
            C_reference[m * ldc + n] = XDNN_BF16(alpha * sum + beta * c_val);
        }
    }

    // Compare results - using higher epsilon due to fp8/bf16 precision
    const float epsilon = 1e-1f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(static_cast<float>(C[i]), static_cast<float>(C_reference[i]), epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] packedB;
    delete[] C;
    delete[] C_reference;
    delete[] scales;
}

// Test with different shapes, transposition flags, and alpha/beta values
TEST_F(AMXSGEMMTest, TransposedAMXTest) {
    // MUST skip before even defining variables because some AMX functions
    // may be called during initialization/compilation
    SkipIfAMXNotSupported();
    
    // Only continue with test setup if we haven't skipped
    // This prevents crashes when AMX is not available
    if(::testing::Test::IsSkipped()) {
        return;
    }

    const int M = 24;
    const int N = 16;
    const int K = 32;
    const float alpha = 1.5f;
    const float beta = 0.5f;
    const bool transA = true;
    const bool transB = false;
    const int lda = M;  // Transposed dimensions
    const int ldb = N;  
    const int ldc = N;

    // Allocate matrices
    XDNN_BF16* A = new XDNN_BF16[K * M];  // K x M because transposed
    XDNN_BF16* B = new XDNN_BF16[K * N];
    XDNN_BF16* C = new XDNN_BF16[M * N];
    XDNN_BF16* C_reference = new XDNN_BF16[M * N];

    // Initialize matrices
    initializeRandomAMXMatrix(A, K * M);
    initializeRandomAMXMatrix(B, K * N);
    initializeRandomAMXMatrix(C, M * N);  // Non-zero initial C for testing beta
    
    // Copy C for reference
    for (int i = 0; i < M * N; ++i) {
        C_reference[i] = C[i];
    }

    // Call the function to test
    // Pack B matrix first
    int pack_size = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(N, K, 16, 16);
    XDNN_BF16* packedB = new XDNN_BF16[pack_size];
    xdnn_small_amx_sgemm_bf16bf16bf16_packb(transB, N, K, B, ldb, packedB, pack_size);
    
    // Then compute the result
    xdnn_small_amx_sgemm_bf16bf16bf16_compute(M, N, K, A, lda, packedB, K, C, ldc, beta);
    
    // Free packed memory
    delete[] packedB;

    // Compute reference result
    reference_amx_gemm_mixed(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

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
