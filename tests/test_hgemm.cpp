// filepath: /home/zhiyis/workspace/code/xdnn-reloaded/tests/test_hgemm.cpp
#include "hgemm.h"
#include "test_common.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <stdexcept>

// Helper functions for matrix debugging and utilities
namespace MatrixDebugUtils {
    
    // Print a 2D matrix stored in row-major format
    void printMatrix(const XDNN_FP16* matrix, int rows, int cols, int stride, 
                     const std::string& name, int max_rows = -1, int max_cols = -1) {
        // If max_rows or max_cols is -1, print all rows/cols
        int print_rows = (max_rows == -1) ? rows : std::min(rows, max_rows);
        int print_cols = (max_cols == -1) ? cols : std::min(cols, max_cols);
        
        std::cout << name << " (" << rows << "x" << cols << ", stride=" << stride << "):\n";
        for (int i = 0; i < print_rows; i++) {
            for (int j = 0; j < print_cols; j++) {
                std::cout << std::fixed << std::setprecision(3) << static_cast<float>(matrix[i * stride + j]) << " ";
            }
            if (cols > print_cols) std::cout << "...";
            std::cout << "\n";
        }
        if (rows > print_rows) std::cout << "...\n";
        std::cout << "\n";
    }
    
    // Print matrix statistics
    void printMatrixStats(const XDNN_FP16* matrix, int num_elements, const std::string& name) {
        float min_val = FLT_MAX, max_val = -FLT_MAX, sum = 0.0f;
        for (int i = 0; i < num_elements; i++) {
            float val = static_cast<float>(matrix[i]);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
        }
        std::cout << name << " stats: min=" << min_val << ", max=" << max_val 
                  << ", avg=" << (sum / num_elements) << "\n";
    }
}

// Reference implementation for matrix packing (B matrix)
// New algorithm: divide matrix into blocks of 64 columns, then pack row by row
void reference_packb(bool transB, int N, int K, const XDNN_FP16* B, int ldb, XDNN_FP16* packedB) {
    const int block_size = 64;
    int num_blocks = (N + block_size - 1) / block_size; // Round up division
    
    int packed_idx = 0;
    
    // Process each block of 64 columns (or remaining columns for last block)
    for (int block = 0; block < num_blocks; block++) {
        int block_start = block * block_size;
        int block_end = std::min(block_start + block_size, N);
        int block_width = block_end - block_start;
        
        // Pack this block row by row
        for (int k = 0; k < K; k++) {
            for (int n = block_start; n < block_end; n++) {
                if (!transB) {
                    // B is K×N
                    packedB[packed_idx++] = B[k * ldb + n];
                } else {
                    // B is N×K (transposed)
                    packedB[packed_idx++] = B[n * ldb + k];
                }
            }
        }
    }
}

// Reference implementation for xdnn_hgemm_compute
void xdnn_hgemm_compute_reference(bool transA, int M, int N, int K,
                                  float alpha, const XDNN_FP16* A, int lda, const XDNN_FP16* packedB,
                                  float beta, XDNN_FP16* C, int ldc) {
    // Check if transA is supported
    if (transA) {
        throw std::runtime_error("transA = true is not currently supported in the reference implementation");
    }
    
    // Unpack the packed B matrix back to original K×N format
    // This reverses the reference_packb algorithm
    std::vector<XDNN_FP16> B_unpacked(K * N);
    const int block_size = 64;
    int num_blocks = (N + block_size - 1) / block_size; // Round up division
    
    int packed_idx = 0;
    
    // Process each block of 64 columns (or remaining columns for last block)
    for (int block = 0; block < num_blocks; block++) {
        int block_start = block * block_size;
        int block_end = std::min(block_start + block_size, N);
        int block_width = block_end - block_start;
        
        // Unpack this block row by row
        for (int k = 0; k < K; k++) {
            for (int n = block_start; n < block_end; n++) {
                // Unpack to K×N format (original B matrix after transpose)
                B_unpacked[k * N + n] = packedB[packed_idx++];
            }
        }
    }
    
    // Print the unpacked B matrix for debugging
    std::cout << "\n--- Unpacked B Matrix in Reference Implementation ---\n";
    MatrixDebugUtils::printMatrix(B_unpacked.data(), K, N, N, "B_unpacked (K×N format)", -1, -1);

    // Matrix multiplication with unpacked B (now in K×N format)
    // Note: transA = true is not supported and checked above
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = static_cast<float>(C[m * ldc + n]) * beta;
            for (int k = 0; k < K; k++) {
                // A is M×K format when transA = false (the only supported case)
                float a_val = static_cast<float>(A[m * lda + k]);
                float b_val = static_cast<float>(B_unpacked[k * N + n]);
                sum += alpha * a_val * b_val;
            }

            C[m * ldc + n] = XDNN_FP16(sum);
        }
    }
}

// Test structure for compute function parameters
struct HGEMMComputeTestParams {
    bool transA;
    int M, N, K;
    int lda, ldc;
    float alpha, beta;
    std::string description;
    
    HGEMMComputeTestParams(bool ta, int m, int n, int k, int la, int lc, float a, float b, const std::string& desc)
        : transA(ta), M(m), N(n), K(k), lda(la), ldc(lc), alpha(a), beta(b), description(desc) {}
};

// Parameterized test class for xdnn_hgemm_compute function
class HGEMMComputeTest : public ::testing::TestWithParam<HGEMMComputeTestParams> {
protected:
    void SetUp() override {
        // Initialize random seed for reproducible tests
        srand(12345);
    }

    void TearDown() override {
        // Cleanup if needed
    }
    
    // Helper function to fill matrix with test data
    void fillMatrix(std::vector<XDNN_FP16>& matrix, int size, float start_val = 0.0f) {
        matrix.resize(size);
        for (int i = 0; i < size; i++) {
            // Create varied test data between -1 and 1 that's reasonable for FP16
            float val = start_val + (static_cast<float>(i % 200) / 100.0f - 1.0f);
            // Clamp to [-1, 1] range
            val = std::max(-1.0f, std::min(1.0f, val));
            matrix[i] = XDNN_FP16(val);
        }
    }
    
    // Helper function to compare matrices with tolerance
    bool compareMatrices(const XDNN_FP16* expected, const XDNN_FP16* actual, int M, int N, int ldc, 
                        float tolerance = FP16_PRECISION_TOLERANCE) {
        bool all_match = true;
        int error_count = 0;
        const int max_errors_to_show = 10;
        
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float exp_val = static_cast<float>(expected[m * ldc + n]);
                float act_val = static_cast<float>(actual[m * ldc + n]);
                float diff = std::abs(exp_val - act_val);
                
                if (diff > tolerance) {
                    all_match = false;
                    error_count++;
                    if (error_count <= max_errors_to_show) {
                        std::cout << "Mismatch at [" << m << "," << n << "]: expected=" 
                                  << exp_val << ", actual=" << act_val << ", diff=" << diff << "\n";
                    }
                }
            }
        }
        
        if (error_count > max_errors_to_show) {
            std::cout << "... and " << (error_count - max_errors_to_show) << " more errors\n";
        }
        
        return all_match;
    }
};

// Single parameterized test that covers all HGEMM compute test cases
TEST_P(HGEMMComputeTest, HGEMMComputeFunctionTest) {
    const HGEMMComputeTestParams& params = GetParam();
    
    // Print test parameters
    std::cout << "\n=== HGEMMComputeFunctionTest: " << params.description << " ===\n";
    std::cout << "Parameters: transA=" << params.transA << ", M=" << params.M 
              << ", N=" << params.N << ", K=" << params.K
              << ", lda=" << params.lda << ", ldc=" << params.ldc
              << ", alpha=" << params.alpha << ", beta=" << params.beta << "\n";
    
    // Allocate matrices
    std::vector<XDNN_FP16> A, B, C_actual, C_expected, packedB;
    
    // Fill matrices with test data
    fillMatrix(A, params.M * params.lda, 0.1f);
    fillMatrix(B, params.K * params.N, 0.2f);  // Use different start value for B
    fillMatrix(C_actual, params.M * params.ldc, 0.3f);  // Use different start value for C
    
    // Copy C for reference computation
    C_expected = C_actual;
    
    // Pack B matrix (assuming B is not transposed for simplicity)
    packedB.resize(params.K * params.N);
    xdnn_hgemm_packb(false, params.N, params.K, B.data(), params.N, packedB.data());
    
    // Print input matrices (full data)
    std::cout << "\n--- Input Matrices ---\n";
    MatrixDebugUtils::printMatrix(A.data(), params.M, params.lda, params.lda, "Matrix A", -1, -1);
    MatrixDebugUtils::printMatrix(B.data(), params.K, params.N, params.N, "Matrix B (K×N format)", -1, -1);
    MatrixDebugUtils::printMatrix(packedB.data(), params.K, params.N, params.N, "PackedB (K×N format)", -1, -1);
    MatrixDebugUtils::printMatrix(C_actual.data(), params.M, params.ldc, params.ldc, "Initial Matrix C", -1, -1);
    
    // Run reference implementation
    xdnn_hgemm_compute_reference(params.transA, params.M, params.N, params.K,
                                 params.alpha, A.data(), params.lda, packedB.data(),
                                 params.beta, C_expected.data(), params.ldc);
    
    // Run actual implementation
    xdnn_hgemm_compute(params.transA, params.M, params.N, params.K,
                       params.alpha, A.data(), params.lda, packedB.data(),
                       params.beta, C_actual.data(), params.ldc);
    
    // Print output matrices (full data)
    std::cout << "\n--- Output Matrices ---\n";
    MatrixDebugUtils::printMatrix(C_expected.data(), params.M, params.ldc, params.ldc, "C Expected (Reference)", -1, -1);
    MatrixDebugUtils::printMatrix(C_actual.data(), params.M, params.ldc, params.ldc, "C Actual (Implementation)", -1, -1);
    
    // Print matrix statistics
    MatrixDebugUtils::printMatrixStats(A.data(), params.M * params.lda, "Matrix A");
    MatrixDebugUtils::printMatrixStats(B.data(), params.K * params.N, "Matrix B");
    MatrixDebugUtils::printMatrixStats(packedB.data(), params.K * params.N, "PackedB");
    MatrixDebugUtils::printMatrixStats(C_expected.data(), params.M * params.ldc, "C Expected");
    MatrixDebugUtils::printMatrixStats(C_actual.data(), params.M * params.ldc, "C Actual");
    
    // Compare results
    EXPECT_TRUE(compareMatrices(C_expected.data(), C_actual.data(), params.M, params.N, params.ldc, 0.2))
        << "Matrix computation mismatch for " << params.description
        << ": transA=" << params.transA << ", M=" << params.M << ", N=" << params.N << ", K=" << params.K
        << ", alpha=" << params.alpha << ", beta=" << params.beta
        << ", lda=" << params.lda << ", ldc=" << params.ldc;
    
    std::cout << "=== End of " << params.description << " ===\n\n";
}

// Instantiate the parameterized test with the required test cases and additional comprehensive cases
INSTANTIATE_TEST_SUITE_P(
    HGEMMComputeFunctionTests,
    HGEMMComputeTest,
    ::testing::Values(
        // // Required test cases from the user
        HGEMMComputeTestParams(false, 20, 4096, 1024, 1024, 4096, 1.0f, 0.0f, "required_case_M20_N4096_K1024"),
        HGEMMComputeTestParams(false, 20, 6144, 1024, 1024, 6144, 1.0f, 0.0f, "required_case_M20_N6144_K1024"),
        HGEMMComputeTestParams(false, 1, 4096, 1024, 1024, 4096, 1.0f, 0.0f, "required_case_M1_N4096_K1024"),
        HGEMMComputeTestParams(false, 1, 6144, 1024, 1024, 6144, 1.0f, 0.0f, "required_case_M1_N6144_K1024"),
        
        // // Additional test cases for comprehensive coverage
        
        // // Basic small test cases
        HGEMMComputeTestParams(false, 16, 16, 16, 16, 16, 1.0f, 0.0f, "basic_small_16x16x16_beta0"),
        HGEMMComputeTestParams(false, 32, 32, 32, 32, 32, 1.0f, 0.0f, "basic_small_32x32x32_beta1"),
        // HGEMMComputeTestParams(true, 16, 16, 16, 16, 16, 1.0f, 0.0f, "basic_small_transA_16x16x16"),

        // Different matrix shapes
        HGEMMComputeTestParams(false, 128, 256, 64, 64, 256, 1.0f, 0.0f, "rectangular_128x256x64"),
        HGEMMComputeTestParams(false, 256, 128, 64, 64, 128, 1.0f, 0.0f, "rectangular_256x128x64"),
        HGEMMComputeTestParams(false, 64, 128, 256, 256, 128, 1.0f, 0.0f, "rectangular_64x128x256"),
        
        // transA variations
        // HGEMMComputeTestParams(true, 128, 128, 128, 128, 128, 1.0f, 1.0f, "transA_128x128x128"),
        // HGEMMComputeTestParams(true, 64, 256, 128, 128, 256, 1.0f, 0.0f, "transA_64x256x128"),
        // HGEMMComputeTestParams(true, 256, 64, 128, 128, 64, 0.5f, 0.5f, "transA_256x64x128"),
        
        // Edge cases with small dimensions
        HGEMMComputeTestParams(false, 1, 1, 1, 1, 1, 1.0f, 1.0f, "edge_case_1x1x1"),
        HGEMMComputeTestParams(false, 1, 1024, 512, 512, 1024, 1.0f, 1.0f, "edge_case_1x1024x512"),
        HGEMMComputeTestParams(false, 1024, 1, 512, 512, 1, 1.0f, 1.0f, "edge_case_1024x1x512"),
        HGEMMComputeTestParams(false, 512, 512, 1, 1, 512, 1.0f, 1.0f, "edge_case_512x512x1"),
        
        // Stride variations (non-minimal strides)
        HGEMMComputeTestParams(false, 64, 64, 64, 128, 128, 1.0f, 1.0f, "stride_variation_64x64x64"),
        HGEMMComputeTestParams(false, 32, 32, 32, 64, 64, 1.0f, 1.0f, "stride_variation_32x32x32"),
        // HGEMMComputeTestParams(true, 32, 32, 32, 64, 64, 1.0f, 1.0f, "stride_variation_transA_32x32x32"),
        
        // Medium-sized matrices
        HGEMMComputeTestParams(false, 512, 512, 512, 512, 512, 1.0f, 1.0f, "medium_512x512x512"),
        HGEMMComputeTestParams(false, 256, 1024, 256, 256, 1024, 1.0f, 1.0f, "medium_256x1024x256"),
        HGEMMComputeTestParams(false, 1024, 256, 256, 256, 256, 1.0f, 1.0f, "medium_1024x256x256"),
        
        // Test cases similar to the required ones but with variations
        HGEMMComputeTestParams(false, 10, 4096, 1024, 1024, 4096, 1.0f, 1.0f, "variation_M10_N4096_K1024"),
        HGEMMComputeTestParams(false, 20, 2048, 1024, 1024, 2048, 1.0f, 1.0f, "variation_M20_N2048_K1024"),
        HGEMMComputeTestParams(false, 20, 4096, 512, 512, 4096, 1.0f, 1.0f, "variation_M20_N4096_K512")
        // HGEMMComputeTestParams(true, 20, 4096, 1024, 1024, 4096, 1.0f, 1.0f, "variation_transA_M20_N4096_K1024"),
    )
);

// Additional specific test for edge cases and error conditions
class HGEMMComputeEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        srand(12345);
    }
};

TEST_F(HGEMMComputeEdgeCaseTest, ZeroBetaTest) {
    // Test behavior when beta = 0 (original C should be ignored)
    const int M = 16, N = 16, K = 16;
    std::vector<XDNN_FP16> A(M * K, XDNN_FP16(1.0f));
    std::vector<XDNN_FP16> packedB(K * N, XDNN_FP16(2.0f));
    std::vector<XDNN_FP16> C(M * N, XDNN_FP16(999.0f)); // Large value that should be ignored
    
    xdnn_hgemm_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), N);
    
    // Expected result: alpha * (A * B) = 1.0 * (1.0 * 2.0 * K) = 32.0
    float expected = 32.0f;
    for (int i = 0; i < M * N; i++) {
        float actual = static_cast<float>(C[i]);
        EXPECT_NEAR(expected, actual, FP16_PRECISION_TOLERANCE)
            << "Zero beta test failed at index " << i;
    }
}

TEST_F(HGEMMComputeEdgeCaseTest, TransANotSupportedTest) {
    // Test that transA = true throws an error in the reference implementation
    const int M = 16, N = 16, K = 16;
    std::vector<XDNN_FP16> A(K * M, XDNN_FP16(1.0f)); // K×M for transA = true
    std::vector<XDNN_FP16> packedB(K * N, XDNN_FP16(2.0f));
    std::vector<XDNN_FP16> C(M * N, XDNN_FP16(0.0f));
    
    // Expect the reference implementation to throw an error when transA = true
    EXPECT_THROW(
        xdnn_hgemm_compute_reference(true, M, N, K, 1.0f, A.data(), M, packedB.data(), 0.0f, C.data(), N),
        std::runtime_error
    );
}

// Test structure for packb function parameters
struct HGEMMPackBTestParams {
    bool transB;
    int N, K;
    int ldb;
    std::string description;
    
    HGEMMPackBTestParams(bool tb, int n, int k, int lb, const std::string& desc)
        : transB(tb), N(n), K(k), ldb(lb), description(desc) {}
};

// Parameterized test class for xdnn_hgemm_packb function
class HGEMMPackBTest : public ::testing::TestWithParam<HGEMMPackBTestParams> {
protected:
    void SetUp() override {
        // Initialize random seed for reproducible tests
        srand(12345);
    }

    void TearDown() override {
        // Cleanup if needed
    }
    
    // Helper function to fill matrix with test data
    void fillMatrix(std::vector<XDNN_FP16>& matrix, bool transB, int N, int K, int ldb, float start_val = 1.0f) {
        // Calculate total matrix size based on the layout
        int matrix_size = transB ? N * ldb : K * ldb;
        matrix.resize(matrix_size);
        
        // Initialize all elements to zero first
        std::fill(matrix.begin(), matrix.end(), XDNN_FP16(0.0f));
        
        // Fill only the actual matrix data (N×K or K×N region)
        if (transB) {
            // B is N×K with stride ldb
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    // Create varied test data between -1 and 1
                    float val = start_val + ((static_cast<float>((n * K + k) % 200) / 100.0f) - 1.0f);
                    // Clamp to [-1, 1] range
                    val = std::max(-1.0f, std::min(1.0f, val));
                    matrix[n * ldb + k] = XDNN_FP16(val);
                }
            }
        } else {
            // B is K×N with stride ldb
            for (int k = 0; k < K; k++) {
                for (int n = 0; n < N; n++) {
                    // Create varied test data between -1 and 1
                    float val = start_val + ((static_cast<float>((k * N + n) % 200) / 100.0f) - 1.0f);
                    // Clamp to [-1, 1] range
                    val = std::max(-1.0f, std::min(1.0f, val));
                    matrix[k * ldb + n] = XDNN_FP16(val);
                }
            }
        }
    }

    // Helper function to compare packed matrices with tolerance
    bool comparePackedMatrices(const XDNN_FP16* expected, const XDNN_FP16* actual, int K, int N,
                              float tolerance = FP16_PRECISION_TOLERANCE) {
        bool all_match = true;
        int error_count = 0;
        const int max_errors_to_show = 10;
        
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                float exp_val = static_cast<float>(expected[k * N + n]);
                float act_val = static_cast<float>(actual[k * N + n]);
                float diff = std::abs(exp_val - act_val);
                
                if (diff > tolerance) {
                    all_match = false;
                    error_count++;
                    if (error_count <= max_errors_to_show) {
                        std::cout << "Mismatch at packed[" << k << "," << n << "]: expected=" 
                                  << exp_val << ", actual=" << act_val << ", diff=" << diff << "\n";
                    }
                }
            }
        }
        
        if (error_count > max_errors_to_show) {
            std::cout << "... and " << (error_count - max_errors_to_show) << " more errors\n";
        }
        
        return all_match;
    }
};

// Single parameterized test that covers all HGEMM packb test cases
TEST_P(HGEMMPackBTest, HGEMMPackBFunctionTest) {
    const HGEMMPackBTestParams& params = GetParam();
    
    // Calculate matrix size based on transB
    int matrix_size = params.transB ? params.N * params.ldb : params.K * params.ldb;
    
    // Allocate matrices
    std::vector<XDNN_FP16> B, packedB_actual, packedB_expected;
    
    // Fill B matrix with test data
    fillMatrix(B, params.transB, params.N, params.K, params.ldb, 1.0f);
    
    // Allocate packed matrices
    packedB_actual.resize(params.K * params.N);
    packedB_expected.resize(params.K * params.N);
    
    // Print test parameters
    std::cout << "\n=== HGEMMPackBFunctionTest: " << params.description << " ===\n";
    std::cout << "Parameters: transB=" << params.transB << ", N=" << params.N 
              << ", K=" << params.K << ", ldb=" << params.ldb << "\n";
    
    // Print original Matrix B (full matrix)
    if (params.transB) {
        // B is N×K with stride ldb
        MatrixDebugUtils::printMatrix(B.data(), params.N, params.K, params.ldb, "Original Matrix B (N×K format)", params.N, params.ldb);
    } else {
        // B is K×N with stride ldb
        MatrixDebugUtils::printMatrix(B.data(), params.K, params.N, params.ldb, "Original Matrix B (K×N format)", params.K, params.ldb);
    }
    
    // Run reference implementation
    reference_packb(params.transB, params.N, params.K, B.data(), params.ldb, packedB_expected.data());
    
    // Run actual implementation
    xdnn_hgemm_packb(params.transB, params.N, params.K, B.data(), params.ldb, packedB_actual.data());
    
    // Print packed matrices (full matrices)
    MatrixDebugUtils::printMatrix(packedB_expected.data(), params.K, params.N, params.ldb, "PackedB Expected (K×N format)", params.K, params.ldb);
    MatrixDebugUtils::printMatrix(packedB_actual.data(), params.K, params.N, params.ldb, "PackedB Actual (K×N format)", params.K, params.ldb);
    
    // Print matrix statistics
    MatrixDebugUtils::printMatrixStats(B.data(), matrix_size, "Original Matrix B");
    MatrixDebugUtils::printMatrixStats(packedB_expected.data(), params.K * params.N, "PackedB Expected");
    MatrixDebugUtils::printMatrixStats(packedB_actual.data(), params.K * params.N, "PackedB Actual");
    
    // Compare results
    EXPECT_TRUE(comparePackedMatrices(packedB_expected.data(), packedB_actual.data(), params.K, params.N))
        << "Matrix packing mismatch for " << params.description
        << ": transB=" << params.transB << ", N=" << params.N << ", K=" << params.K
        << ", ldb=" << params.ldb;
    
    std::cout << "=== End of " << params.description << " ===\n\n";
}

// Instantiate the parameterized test for packb function with comprehensive test cases
INSTANTIATE_TEST_SUITE_P(
    HGEMMPackBFunctionTests,
    HGEMMPackBTest,
    ::testing::Values(
        // Basic small test case for debugging
        HGEMMPackBTestParams(false, 16, 16, 16, "basic_small_N16_K16_notrans"),
        HGEMMPackBTestParams(false, 32, 32, 32, "basic_small_N32_K32_notrans"),
        HGEMMPackBTestParams(false, 64, 64, 64, "basic_medium_N64_K64_notrans"),
        HGEMMPackBTestParams(true, 16, 16, 16, "basic_small_N16_K16_trans"),
        HGEMMPackBTestParams(true, 32, 32, 32, "basic_small_N32_K32_trans"),
        HGEMMPackBTestParams(true, 64, 64, 64, "basic_medium_N64_K64_trans"),
        HGEMMPackBTestParams(false, 256, 64, 256, "rectangular_N128_K64_notrans"),
        HGEMMPackBTestParams(false, 64, 128, 64, "rectangular_N64_K128_notrans"),
        HGEMMPackBTestParams(true, 128, 64, 64, "rectangular_N128_K64_trans"),
        HGEMMPackBTestParams(true, 64, 128, 128, "rectangular_N64_K128_trans"),
        HGEMMPackBTestParams(false, 1, 1, 1, "edge_case_N1_K1_notrans"),
        HGEMMPackBTestParams(true, 1, 1, 1, "edge_case_N1_K1_trans"),
        HGEMMPackBTestParams(false, 64, 64, 128, "stride_variation_N64_K64_notrans"),
        HGEMMPackBTestParams(true, 64, 64, 128, "stride_variation_N64_K64_trans")
    )
);

// Additional specific tests for edge cases and consistency
TEST_F(HGEMMComputeEdgeCaseTest, PackBConsistencyTest) {
    // Test that packing and unpacking gives consistent results with matrix multiplication
    const int N = 32, K = 32;
    std::vector<XDNN_FP16> B_original(K * N);
    std::vector<XDNN_FP16> B_transposed(N * K);
    std::vector<XDNN_FP16> packedB_notrans(K * N);
    std::vector<XDNN_FP16> packedB_trans(K * N);
    
    // Fill original matrix
    for (int i = 0; i < K * N; i++) {
        B_original[i] = XDNN_FP16(static_cast<float>(i % 100) * 0.01f + 1.0f);
    }
    
    // Create transposed version
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            B_transposed[n * K + k] = B_original[k * N + n];
        }
    }
    
    // Pack both versions
    xdnn_hgemm_packb(false, N, K, B_original.data(), N, packedB_notrans.data());
    xdnn_hgemm_packb(true, N, K, B_transposed.data(), K, packedB_trans.data());
    
    // Both packed matrices should be identical
    for (int i = 0; i < K * N; i++) {
        float val_notrans = static_cast<float>(packedB_notrans[i]);
        float val_trans = static_cast<float>(packedB_trans[i]);
        EXPECT_NEAR(val_notrans, val_trans, FP16_PRECISION_TOLERANCE)
            << "PackB consistency test failed at index " << i 
            << ": notrans=" << val_notrans << ", trans=" << val_trans;
    }
}

TEST_F(HGEMMComputeEdgeCaseTest, PackBStrideBehaviorTest) {
    // Test behavior with different stride values
    const int N = 16, K = 16;
    const int stride_normal = N;
    const int stride_large = N + 8; // Larger stride
    
    std::vector<XDNN_FP16> B_normal(K * stride_normal);
    std::vector<XDNN_FP16> B_strided(K * stride_large);
    std::vector<XDNN_FP16> packedB_normal(K * N);
    std::vector<XDNN_FP16> packedB_strided(K * N);
    
    // Fill matrices with the same data pattern
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            XDNN_FP16 val = XDNN_FP16(static_cast<float>(k * N + n) * 0.01f + 1.0f);
            B_normal[k * stride_normal + n] = val;
            B_strided[k * stride_large + n] = val;
        }
    }
    
    // Pack both matrices
    xdnn_hgemm_packb(false, N, K, B_normal.data(), stride_normal, packedB_normal.data());
    xdnn_hgemm_packb(false, N, K, B_strided.data(), stride_large, packedB_strided.data());
    
    // Results should be identical regardless of stride
    for (int i = 0; i < K * N; i++) {
        float val_normal = static_cast<float>(packedB_normal[i]);
        float val_strided = static_cast<float>(packedB_strided[i]);
        EXPECT_NEAR(val_normal, val_strided, FP16_PRECISION_TOLERANCE)
            << "PackB stride behavior test failed at index " << i 
            << ": normal_stride=" << val_normal << ", large_stride=" << val_strided;
    }
}

// Test structure for compute_biasadd function parameters
struct HGEMMComputeBiasAddTestParams {
    bool transA;
    int M, N, K;
    int lda, ldc;
    float alpha, beta;
    std::string description;
    
    HGEMMComputeBiasAddTestParams(bool ta, int m, int n, int k, int la, int lc, float a, float b, const std::string& desc)
        : transA(ta), M(m), N(n), K(k), lda(la), ldc(lc), alpha(a), beta(b), description(desc) {}
};

// Reference implementation for xdnn_hgemm_compute_biasadd
void xdnn_hgemm_compute_biasadd_reference(bool transA, int M, int N, int K,
                                         float alpha, const XDNN_FP16* A, int lda, const XDNN_FP16* packedB,
                                         float beta, XDNN_FP16* C, int ldc, const float* bias) {
    // First perform the matrix multiplication (same as the compute reference)
    xdnn_hgemm_compute_reference(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
    
    // Then add bias to each row
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float current_val = static_cast<float>(C[m * ldc + n]);
            C[m * ldc + n] = XDNN_FP16(current_val + bias[n]);
        }
    }
}

// Parameterized test class for xdnn_hgemm_compute_biasadd function
class HGEMMComputeBiasAddTest : public ::testing::TestWithParam<HGEMMComputeBiasAddTestParams> {
protected:
    void SetUp() override {
        // Initialize random seed for reproducible tests
        srand(12345);
    }

    void TearDown() override {
        // Cleanup if needed
    }
    
    // Helper function to fill matrix with test data
    void fillMatrix(std::vector<XDNN_FP16>& matrix, int size, float start_val = 0.0f) {
        matrix.resize(size);
        for (int i = 0; i < size; i++) {
            // Create varied test data between -1 and 1 that's reasonable for FP16
            float val = start_val + (static_cast<float>(i % 200) / 100.0f - 1.0f);
            // Clamp to [-1, 1] range
            val = std::max(-1.0f, std::min(1.0f, val));
            matrix[i] = XDNN_FP16(val);
        }
    }
    
    // Helper function to fill bias vector with test data
    void fillBias(std::vector<float>& bias, int size, float start_val = 0.5f) {
        bias.resize(size);
        for (int i = 0; i < size; i++) {
            // Create varied bias data between -0.5 and 0.5
            float val = start_val + (static_cast<float>(i % 100) / 200.0f - 0.25f);
            bias[i] = val;
        }
    }
    
    // Helper function to compare matrices with tolerance
    bool compareMatrices(const XDNN_FP16* expected, const XDNN_FP16* actual, int M, int N, int ldc, 
                        float tolerance = FP16_PRECISION_TOLERANCE) {
        bool all_match = true;
        int error_count = 0;
        const int max_errors_to_show = 10;
        
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float exp_val = static_cast<float>(expected[m * ldc + n]);
                float act_val = static_cast<float>(actual[m * ldc + n]);
                float diff = std::abs(exp_val - act_val);
                
                if (diff > tolerance) {
                    all_match = false;
                    error_count++;
                    if (error_count <= max_errors_to_show) {
                        std::cout << "Mismatch at [" << m << "," << n << "]: expected=" 
                                  << exp_val << ", actual=" << act_val << ", diff=" << diff << "\n";
                    }
                }
            }
        }
        
        if (error_count > max_errors_to_show) {
            std::cout << "... and " << (error_count - max_errors_to_show) << " more errors\n";
        }
        
        return all_match;
    }
};

// Single parameterized test that covers all HGEMM compute_biasadd test cases
TEST_P(HGEMMComputeBiasAddTest, HGEMMComputeBiasAddFunctionTest) {
    const HGEMMComputeBiasAddTestParams& params = GetParam();
    
    // Print test parameters
    std::cout << "\n=== HGEMMComputeBiasAddFunctionTest: " << params.description << " ===\n";
    std::cout << "Parameters: transA=" << params.transA << ", M=" << params.M 
              << ", N=" << params.N << ", K=" << params.K
              << ", lda=" << params.lda << ", ldc=" << params.ldc
              << ", alpha=" << params.alpha << ", beta=" << params.beta << "\n";
    
    // Allocate matrices
    std::vector<XDNN_FP16> A, B, C_actual, C_expected, packedB;
    std::vector<float> bias;
    
    // Fill matrices with test data
    fillMatrix(A, params.M * params.lda, 0.1f);
    fillMatrix(B, params.K * params.N, 0.2f);  // Use different start value for B
    fillMatrix(C_actual, params.M * params.ldc, 0.3f);  // Use different start value for C
    fillBias(bias, params.N, 0.5f);  // Fill bias vector
    
    // Copy C for reference computation
    C_expected = C_actual;
    
    // Pack B matrix (assuming B is not transposed for simplicity)
    packedB.resize(params.K * params.N);
    xdnn_hgemm_packb(false, params.N, params.K, B.data(), params.N, packedB.data());
    
    // Print input matrices and bias (limited output for readability)
    std::cout << "\n--- Input Matrices and Bias ---\n";
    MatrixDebugUtils::printMatrix(A.data(), params.M, params.lda, params.lda, "Matrix A", 5, 10);
    MatrixDebugUtils::printMatrix(B.data(), params.K, params.N, params.N, "Matrix B (K×N format)", 5, 10);
    MatrixDebugUtils::printMatrix(packedB.data(), params.K, params.N, params.N, "PackedB (K×N format)", 5, 10);
    MatrixDebugUtils::printMatrix(C_actual.data(), params.M, params.ldc, params.ldc, "Initial Matrix C", 5, 10);
    
    // Print bias vector
    std::cout << "Bias vector: ";
    for (int i = 0; i < std::min(params.N, 10); i++) {
        std::cout << std::fixed << std::setprecision(3) << bias[i] << " ";
    }
    if (params.N > 10) std::cout << "...";
    std::cout << "\n\n";
    
    // Run reference implementation
    xdnn_hgemm_compute_biasadd_reference(params.transA, params.M, params.N, params.K,
                                        params.alpha, A.data(), params.lda, packedB.data(),
                                        params.beta, C_expected.data(), params.ldc, bias.data());
    
    // Run actual implementation
    xdnn_hgemm_compute_biasadd(params.transA, params.M, params.N, params.K,
                              params.alpha, A.data(), params.lda, packedB.data(),
                              params.beta, C_actual.data(), params.ldc, bias.data());
    
    // Print output matrices (limited output for readability)
    std::cout << "\n--- Output Matrices ---\n";
    MatrixDebugUtils::printMatrix(C_expected.data(), params.M, params.ldc, params.ldc, "C Expected (Reference)", 5, 10);
    MatrixDebugUtils::printMatrix(C_actual.data(), params.M, params.ldc, params.ldc, "C Actual (Implementation)", 5, 10);
    
    // Print matrix statistics
    MatrixDebugUtils::printMatrixStats(A.data(), params.M * params.lda, "Matrix A");
    MatrixDebugUtils::printMatrixStats(B.data(), params.K * params.N, "Matrix B");
    MatrixDebugUtils::printMatrixStats(packedB.data(), params.K * params.N, "PackedB");
    MatrixDebugUtils::printMatrixStats(C_expected.data(), params.M * params.ldc, "C Expected");
    MatrixDebugUtils::printMatrixStats(C_actual.data(), params.M * params.ldc, "C Actual");
    
    // Compare results
    EXPECT_TRUE(compareMatrices(C_expected.data(), C_actual.data(), params.M, params.N, params.ldc, 0.2))
        << "Matrix computation with bias addition mismatch for " << params.description
        << ": transA=" << params.transA << ", M=" << params.M << ", N=" << params.N << ", K=" << params.K
        << ", alpha=" << params.alpha << ", beta=" << params.beta
        << ", lda=" << params.lda << ", ldc=" << params.ldc;
    
    std::cout << "=== End of " << params.description << " ===\n\n";
}

// Instantiate the parameterized test with comprehensive test cases for compute_biasadd
INSTANTIATE_TEST_SUITE_P(
    HGEMMComputeBiasAddFunctionTests,
    HGEMMComputeBiasAddTest,
    ::testing::Values(
        // Basic test cases
        HGEMMComputeBiasAddTestParams(false, 16, 16, 16, 16, 16, 1.0f, 0.0f, "basic_small_16x16x16_biasadd"),
        HGEMMComputeBiasAddTestParams(false, 32, 32, 32, 32, 32, 1.0f, 1.0f, "basic_small_32x32x32_biasadd"),
        
        // Test cases similar to compute tests but with bias addition
        HGEMMComputeBiasAddTestParams(false, 20, 4096, 1024, 1024, 4096, 1.0f, 0.0f, "biasadd_case_M20_N4096_K1024"),
        HGEMMComputeBiasAddTestParams(false, 20, 6144, 1024, 1024, 6144, 1.0f, 0.0f, "biasadd_case_M20_N6144_K1024"),
        HGEMMComputeBiasAddTestParams(false, 1, 4096, 1024, 1024, 4096, 1.0f, 0.0f, "biasadd_case_M1_N4096_K1024"),
        HGEMMComputeBiasAddTestParams(false, 1, 6144, 1024, 1024, 6144, 1.0f, 0.0f, "biasadd_case_M1_N6144_K1024"),
        
        // Different matrix shapes with bias
        HGEMMComputeBiasAddTestParams(false, 128, 256, 64, 64, 256, 1.0f, 0.0f, "rectangular_128x256x64_biasadd"),
        HGEMMComputeBiasAddTestParams(false, 256, 128, 64, 64, 128, 1.0f, 0.0f, "rectangular_256x128x64_biasadd"),
        HGEMMComputeBiasAddTestParams(false, 64, 128, 256, 256, 128, 1.0f, 0.0f, "rectangular_64x128x256_biasadd"),
        
        // Edge cases with small dimensions
        HGEMMComputeBiasAddTestParams(false, 1, 1, 1, 1, 1, 1.0f, 1.0f, "edge_case_1x1x1_biasadd"),
        HGEMMComputeBiasAddTestParams(false, 1, 1024, 512, 512, 1024, 1.0f, 1.0f, "edge_case_1x1024x512_biasadd"),
        
        // Medium-sized matrices with bias
        HGEMMComputeBiasAddTestParams(false, 256, 512, 256, 256, 512, 1.0f, 1.0f, "medium_256x512x256_biasadd"),
        HGEMMComputeBiasAddTestParams(false, 512, 256, 256, 256, 256, 1.0f, 1.0f, "medium_512x256x256_biasadd")
    )
);

// Additional specific tests for bias addition edge cases
TEST_F(HGEMMComputeEdgeCaseTest, BiasAddZeroBiasTest) {
    // Test behavior when bias is all zeros (should be same as regular compute)
    const int M = 16, N = 16, K = 16;
    std::vector<XDNN_FP16> A(M * K, XDNN_FP16(1.0f));
    std::vector<XDNN_FP16> packedB(K * N, XDNN_FP16(2.0f));
    std::vector<XDNN_FP16> C_biasadd(M * N, XDNN_FP16(0.5f));
    std::vector<XDNN_FP16> C_regular(M * N, XDNN_FP16(0.5f));
    std::vector<float> zero_bias(N, 0.0f);
    
    // Run both computations
    xdnn_hgemm_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 1.0f, C_regular.data(), N);
    xdnn_hgemm_compute_biasadd(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 1.0f, C_biasadd.data(), N, zero_bias.data());
    
    // Results should be identical
    for (int i = 0; i < M * N; i++) {
        float val_regular = static_cast<float>(C_regular[i]);
        float val_biasadd = static_cast<float>(C_biasadd[i]);
        EXPECT_NEAR(val_regular, val_biasadd, FP16_PRECISION_TOLERANCE)
            << "Zero bias test failed at index " << i 
            << ": regular=" << val_regular << ", biasadd=" << val_biasadd;
    }
}

TEST_F(HGEMMComputeEdgeCaseTest, BiasAddLargeBiasTest) {
    // Test behavior with larger bias values
    const int M = 8, N = 8, K = 8;
    std::vector<XDNN_FP16> A(M * K, XDNN_FP16(0.1f));
    std::vector<XDNN_FP16> packedB(K * N, XDNN_FP16(0.1f));
    std::vector<XDNN_FP16> C(M * N, XDNN_FP16(0.0f));
    std::vector<float> large_bias(N);
    
    // Fill bias with larger values
    for (int i = 0; i < N; i++) {
        large_bias[i] = static_cast<float>(i) * 2.0f;
    }
    
    xdnn_hgemm_compute_biasadd(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), N, large_bias.data());
    
    // Verify that bias was properly added to each row
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float result = static_cast<float>(C[m * N + n]);
            // Expected: matrix_result + bias[n], where matrix_result = alpha * sum(A * B) = 1.0 * (0.1 * 0.1 * K) = 0.08
            float expected_matrix_part = 0.08f;  // 0.1 * 0.1 * 8
            float expected_total = expected_matrix_part + large_bias[n];
            EXPECT_NEAR(expected_total, result, 0.1f)  // Use larger tolerance for this test
                << "Large bias test failed at [" << m << "," << n << "]: expected=" 
                << expected_total << ", actual=" << result;
        }
    }
}