#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include "hgemm.h"
#include "hgemm_f16f16f32.h"
#include "hgemm_f32f16f32.h"
#include "hgemm_f32f16f16.h"
#include "hgemm_f32s8f32.h"
#include "hgemm_f32u4f32.h"
#include "data_types/float16.h"
#include "data_types/uint4x2.h"
#include "../platform_detection.h"

// Helper function to initialize random matrices
template<typename T>
void initializeRandomTypedMatrix(T* matrix, size_t size, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    for (size_t i = 0; i < size; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            matrix[i] = dist(gen);
        } else if constexpr (std::is_same_v<T, XDNN_FP16>) {
            matrix[i] = XDNN_FP16(dist(gen));
        } else if constexpr (std::is_same_v<T, int8_t>) {
            matrix[i] = static_cast<int8_t>(dist(gen) * 127);
        } else {
            // Default case for other types
            matrix[i] = static_cast<T>(dist(gen));
        }
    }
}

// Helper function for reference matrix multiplication with mixed types
template<typename TA, typename TB, typename TC>
void reference_gemm_mixed(bool transA, bool transB, int M, int N, int K,
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

// Test fixture for HGEMM tests
class HGEMMTest : public ::testing::Test {
protected:
    xdnn::CPUFeatures cpuFeatures;
    
    void SetUp() override {
        // Detect CPU features for conditional tests
        cpuFeatures = xdnn::detectCPUFeatures();
    }

    void TearDown() override {
        // Common teardown code
    }
    
    // Skip if hardware doesn't support FP16 operations
    void SkipIfFP16NotSupported() {
        // We're using the same check as SkipIfNotXeon for consistency
        bool supportsFP16 = cpuFeatures.supportsAVX512();
        if (!supportsFP16) {
            GTEST_SKIP() << "Skipping test that requires FP16 support which is not available on this CPU";
        }
    }
    
    // Skip if not running on Intel Xeon with AVX512
    void SkipIfNotXeon() {
        bool isXeon = cpuFeatures.supportsAVX512();
        if (!isXeon) {
            GTEST_SKIP() << "Skipping test that requires Intel Xeon platform which is not available";
        }
    }
};

// Test xdnn_hgemm_f16f16f32 function
TEST_F(HGEMMTest, F16F16F32Test) {
    // Skip if hardware doesn't support FP16 operations
    SkipIfFP16NotSupported();
    
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
    XDNN_FP16* A = new XDNN_FP16[M * K];
    XDNN_FP16* B = new XDNN_FP16[K * N];
    float* C = new float[M * N];
    float* C_reference = new float[M * N];

    // Initialize matrices
    initializeRandomTypedMatrix(A, M * K);
    initializeRandomTypedMatrix(B, K * N);
    memset(C, 0, M * N * sizeof(float));
    memset(C_reference, 0, M * N * sizeof(float));

    // Call the function to test
    xdnn_hgemm_f16f16f32(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    // Compute reference result
    reference_gemm_mixed(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    // Compare results
    const float epsilon = 1e-2f;  // Larger epsilon for lower precision computation
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_reference[i], epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
}

// Test xdnn_hgemm_f32f16f32 function
TEST_F(HGEMMTest, F32F16F32Test) {
    // Skip if hardware doesn't support FP16 operations
    SkipIfFP16NotSupported();
    
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
    XDNN_FP16* B = new XDNN_FP16[K * N];
    float* C = new float[M * N];
    float* C_reference = new float[M * N];

    // Initialize matrices
    initializeRandomTypedMatrix(A, M * K);
    initializeRandomTypedMatrix(B, K * N);
    memset(C, 0, M * N * sizeof(float));
    memset(C_reference, 0, M * N * sizeof(float));

    // Call the function to test
    xdnn_hgemm_f32f16f32(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    // Compute reference result
    reference_gemm_mixed(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    // Compare results
    const float epsilon = 1e-2f;  // Larger epsilon for lower precision computation
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_reference[i], epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
}

// Test xdnn_hgemm_f32f16f16 function
TEST_F(HGEMMTest, F32F16F16Test) {
    // Skip if hardware doesn't support FP16 operations
    SkipIfFP16NotSupported();
    
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
    XDNN_FP16* B = new XDNN_FP16[K * N];
    XDNN_FP16* C = new XDNN_FP16[M * N];
    XDNN_FP16* C_reference = new XDNN_FP16[M * N];

    // Initialize matrices
    initializeRandomTypedMatrix(A, M * K);
    initializeRandomTypedMatrix(B, K * N);
    initializeRandomTypedMatrix(C, M * N, 0.0f, 0.0f);  // Initialize to zeros
    initializeRandomTypedMatrix(C_reference, M * N, 0.0f, 0.0f);  // Initialize to zeros

    // Call the function to test
    xdnn_hgemm_f32f16f16(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    // Compute reference result
    reference_gemm_mixed(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    // Compare results - using higher epsilon due to fp16 precision
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

// Test xdnn_hgemm_f32s8f32 function
TEST_F(HGEMMTest, F32S8F32Test) {
    // Skip if not running on Intel Xeon with AVX512
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
    int8_t* B = new int8_t[K * N];
    float* C = new float[M * N];
    float* C_reference = new float[M * N];
    float* scales = new float[N];  // Scales for int8 quantization

    // Initialize matrices
    initializeRandomTypedMatrix(A, M * K);
    initializeRandomTypedMatrix(B, K * N);
    memset(C, 0, M * N * sizeof(float));
    memset(C_reference, 0, M * N * sizeof(float));
    
    // Initialize scales (normally these would be computed from quantization)
    for (int i = 0; i < N; i++) {
        scales[i] = 0.1f;  // Simple scale factor for testing
    }

    // Create zero points array
    float* zeros = new float[N];  // Zero points for int8 quantization
    for (int i = 0; i < N; i++) {
        zeros[i] = 0.0f;  // Assume symmetric quantization for testing
    }

    // Call the function to test
    xdnn_hgemm_f32s8f32(transA, transB, M, N, K, alpha, A, lda, B, ldb, scales, zeros, beta, C, ldc);

    // Compute reference result with scaling
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                float b_val = transB ? 
                    static_cast<float>(B[n * ldb + k]) * scales[n] : 
                    static_cast<float>(B[k * ldb + n]) * scales[n];
                sum += a_val * b_val;
            }
            C_reference[m * ldc + n] = alpha * sum + beta * C_reference[m * ldc + n];
        }
    }

    // Compare results - using higher epsilon due to int8 quantization
    const float epsilon = 1e-1f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_reference[i], epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
    delete[] scales;
    delete[] zeros;
}

// Test xdnn_hgemm_f32u4f32 function
TEST_F(HGEMMTest, F32U4F32Test) {
    // Skip if not running on Intel Xeon with AVX512
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
    const int ldb = N / 2;  // Each XDNN_UINT4x2 contains 2 uint4 values
    const int ldc = N;

    // Allocate matrices
    float* A = new float[M * K];
    XDNN_UINT4x2* B = new XDNN_UINT4x2[(K * N + 1) / 2];  // Each uint4x2 packs 2 values
    float* C = new float[M * N];
    float* C_reference = new float[M * N];
    float* scales = new float[N];  // Scales for uint4 quantization
    int32_t* zeros = new int32_t[N];  // Zero points for uint4 quantization

    // Initialize matrices
    initializeRandomTypedMatrix(A, M * K);
    memset(B, 0, ((K * N + 1) / 2) * sizeof(XDNN_UINT4x2));
    memset(C, 0, M * N * sizeof(float));
    memset(C_reference, 0, M * N * sizeof(float));
    
    // Initialize scales and zero points
    for (int i = 0; i < N; i++) {
        scales[i] = 0.1f;  // Simple scale factor for testing
        zeros[i] = 8;      // Mid-point of uint4 range (0-15)
    }
    
    // Manual initialization of uint4 values
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n += 2) {
            if (n + 1 < N) {
                // Pack two uint4 values into one uint4x2
                uint8_t val1 = (n % 16);        // 0-15 range
                uint8_t val2 = ((n+1) % 16);    // 0-15 range
                B[(k * N + n) / 2] = XDNN_UINT4x2(val1, val2);
            } else {
                // Only one value at the end
                B[(k * N + n) / 2] = XDNN_UINT4x2(n % 16);
            }
        }
    }

    // Prepare zeros as float array
    float* zeros_f = new float[N];
    for (int i = 0; i < N; i++) {
        zeros_f[i] = static_cast<float>(zeros[i]);
    }

    // Call the function to test
    xdnn_hgemm_f32u4f32(transA, transB, M, N, K, alpha, A, lda, B, ldb, scales, zeros_f, beta, C, ldc);

    // Create unpacked B matrix for reference computation
    uint8_t* B_unpacked = new uint8_t[K * N];
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n += 2) {
            if (n + 1 < N) {
                B_unpacked[k * N + n] = B[(k * N + n) / 2].get_v1();
                B_unpacked[k * N + n + 1] = B[(k * N + n) / 2].get_v2();
            } else {
                B_unpacked[k * N + n] = B[(k * N + n) / 2].get_v1();
            }
        }
    }

    // Compute reference result with dequantization
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                
                uint8_t b_quant_val = transB ? 
                    B_unpacked[n * K + k] : 
                    B_unpacked[k * N + n];
                    
                // Dequantize: scale * (value - zero_point)
                float b_val = scales[n] * (static_cast<float>(b_quant_val) - static_cast<float>(zeros[n]));
                
                sum += a_val * b_val;
            }
            C_reference[m * ldc + n] = alpha * sum + beta * C_reference[m * ldc + n];
        }
    }

    // Compare results - using higher epsilon due to uint4 quantization
    const float epsilon = 1e-1f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_reference[i], epsilon);
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_reference;
    delete[] scales;
    delete[] zeros;
    delete[] zeros_f;
    delete[] B_unpacked;
}
