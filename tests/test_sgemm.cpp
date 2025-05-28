#include "gtest/gtest.h"
#include "sgemm.h" // Should be found via include_directories
#include "data_types/data_types.h" // For XDNN_BF16 etc., though sgemm.h is float only
#include "test_common.h" // For potential future common utilities, though not strictly needed now

#include <vector>
#include <numeric> // For std::iota if used for initialization
#include <cmath>     // For std::tanh, std::exp, std::pow, std::sqrt, std::abs, atan
#include <algorithm> // For std::max, std::fill, std::generate
#include <random>    // For random data initialization

#ifndef M_PI
#define M_PI (atan(1.0f) * 4.0f)
#endif

// Helper function to initialize a matrix with sequential or random values
void init_matrix(std::vector<float>& matrix, int rows, int cols, int ld, bool sequential = true, int seed_offset = 0) {
    matrix.resize(ld * (rows > 0 ? rows : 1)); // Ensure enough space even if one dim is 0 for safety, ld * rows for typical case
    if (rows == 0 || cols == 0) return; // Nothing to initialize for empty matrix

    if (sequential) {
        float val = 1.0f;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                matrix[r * ld + c] = val++;
            }
        }
    } else {
        std::mt19937 gen(1337 + seed_offset); // Basic seed
        std::uniform_real_distribution<float> distrib(-2.0f, 2.0f);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                matrix[r * ld + c] = distrib(gen);
            }
        }
    }
}

// Reference SGEMM computation: C = alpha * A * B + beta * C
void reference_sgemm_computation(
    bool transA, bool transB, int M, int N, int K,
    float alpha, const std::vector<float>& A, int lda,
    const std::vector<float>& B, int ldb,
    float beta, std::vector<float>& C, int ldc) {

    if (M == 0 || N == 0) return; // No computation if output is empty

    // Check alpha and beta for xdnn_sgemm compatibility
    if (alpha != 1.0f || (beta != 0.0f && beta != 1.0f)) {
        std::cout << "[SKIP] reference_sgemm_computation: Only alpha=1.0f and beta=0.0f or 1.0f are supported for xdnn_sgemm tests. Got alpha=" << alpha << ", beta=" << beta << std::endl;
        GTEST_SKIP() << "reference_sgemm_computation: Only alpha=1.0f and beta=0.0f or 1.0f are supported for xdnn_sgemm tests. Got alpha=" << alpha << ", beta=" << beta;
        return;
    }

    // If beta is 0, C should be initialized to 0 before accumulation.
    // If beta is not 0, C is an input. The test setup should pre-initialize C.
    // For simplicity here, if beta is 0, we'll scale by 0. If beta is non-zero, C is used as input.
    // The test harness calling this should ensure C is correctly sized and initialized.

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            if (K > 0) { // Only compute sum if K > 0
                for (int k = 0; k < K; ++k) {
                    float valA = transA ? A[k * lda + m] : A[m * lda + k];
                    float valB = transB ? B[n * ldb + k] : B[k * ldb + n];
                    sum += valA * valB;
                }
            }
            
            if (beta == 0.0f) {
                 C[m * ldc + n] = alpha * sum;
            } else {
                 C[m * ldc + n] = alpha * sum + beta * C[m * ldc + n];
            }
        }
    }
}

// Helper reference functions for post-operations

inline float sigmoid_op(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float silu_op(float x) {
    return x * sigmoid_op(x);
}

void apply_silu_to_matrix(std::vector<float>& matrix, int M, int N, int ldc) {
    if (M == 0 || N == 0) return;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            matrix[r * ldc + c] = silu_op(matrix[r * ldc + c]);
        }
    }
}

inline float gelu_approx_op(float x) {
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3.0f))));
}

void apply_gelu_to_matrix(std::vector<float>& matrix, int M, int N, int ldc) {
    if (M == 0 || N == 0) return;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            matrix[r * ldc + c] = gelu_approx_op(matrix[r * ldc + c]);
        }
    }
}

inline float relu_op(float x) {
    return std::max(0.0f, x);
}

void apply_relu_to_matrix(std::vector<float>& matrix, int M, int N, int ldc) {
    if (M == 0 || N == 0) return;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            matrix[r * ldc + c] = relu_op(matrix[r * ldc + c]);
        }
    }
}

void apply_bias_add_to_matrix(std::vector<float>& C, int M, int N, int ldc, const std::vector<float>& bias_vec) {
    if (M == 0 || N == 0 || bias_vec.empty()) return;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            C[r * ldc + c] += bias_vec[c];
        }
    }
}

void apply_residential_to_matrix(std::vector<float>& C, int M, int N, int ldc, const std::vector<float>& bias_vec, const std::vector<float>& res_vec, int ldres) {
    if (M == 0 || N == 0) return;
    bool has_bias = !bias_vec.empty();
    bool has_res = !res_vec.empty();

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            if (has_bias) C[r * ldc + c] += bias_vec[c];
            if (has_res)  C[r * ldc + c] += res_vec[r * ldres + c];
        }
    }
}

void apply_resext_to_matrix(std::vector<float>& C, int M, int N, int ldc, const std::vector<float>& bias_vec, float gamma, const std::vector<float>& res_vec, int ldres) {
    if (M == 0 || N == 0) return;
    bool has_bias = !bias_vec.empty();
    bool has_res = !res_vec.empty();

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            if (has_bias) C[r * ldc + c] += bias_vec[c];
            if (has_res)  C[r * ldc + c] += gamma * res_vec[r * ldres + c];
        }
    }
}

void apply_resmul_to_matrix(std::vector<float>& C, int M, int N, int ldc, const std::vector<float>& res_vec, int ldres) {
    if (M == 0 || N == 0 || res_vec.empty()) return;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            C[r * ldc + c] *= res_vec[r * ldres + c];
        }
    }
}


// Test Fixture for SGEMM tests
class SgemmTest : public ::testing::Test {
protected:
    std::vector<float> A, B, C_actual, C_expected, packedB_actual;
    // For bias, res etc. in future tests
    std::vector<float> bias, res;

    void verify_matrices(int M, int N, int ldc) {
        ASSERT_EQ(C_actual.size(), C_expected.size());
        if (M == 0 || N == 0) return;
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < N; ++c) {
                EXPECT_NEAR(C_actual[r * ldc + c], C_expected[r * ldc + c], FP32_PRECISION_TOLERANCE)
                    << "Mismatch at C[" << r << "][" << c << "]";
            }
        }
    }
};

// Test parameters: M, N, K, lda_mul, ldb_mul, ldc_mul, transA, transB, alpha, beta
struct SgemmParams {
    int M, N, K;
    float lda_mul, ldb_mul, ldc_mul; // Multipliers for leading dimensions (e.g. 1.0 for tight, 1.5 for padding)
    bool transA, transB;
    float alpha, beta;
};

class SgemmParamTest : public SgemmTest, public ::testing::WithParamInterface<SgemmParams> {};

// New Param Struct and Test Class for ResExt
struct SgemmResExtParams : public SgemmParams {
    float gamma;
};

class SgemmResExtParamTest : public SgemmTest, public ::testing::WithParamInterface<SgemmResExtParams> {};


TEST_P(SgemmParamTest, XdnnSgemm) {
    SgemmParams p = GetParam();
    int lda = static_cast<int>((p.transA ? p.M : p.K) * p.lda_mul);
    if (p.transA && p.K > 0) lda = static_cast<int>(p.M * p.lda_mul); else if (p.M > 0) lda = static_cast<int>(p.K * p.lda_mul);
    if (p.transA) lda = std::max(p.M,1) * p.lda_mul; else lda = std::max(p.K,1) * p.lda_mul;
    if (p.transA) lda = static_cast<int>(std::max(p.M, 1) * p.lda_mul); else lda = static_cast<int>(std::max(p.K, 1) * p.lda_mul);

    int rowsA = p.transA ? p.K : p.M;
    int colsA = p.transA ? p.M : p.K;
    lda = std::max(colsA, static_cast<int>(colsA * p.lda_mul));
    if (rowsA == 0 || colsA == 0) lda = 0;

    int rowsB = p.transB ? p.N : p.K;
    int colsB = p.transB ? p.K : p.N;
    int ldb = std::max(colsB, static_cast<int>(colsB * p.ldb_mul));
    if (rowsB == 0 || colsB == 0) ldb = 0;
    
    int ldc = std::max(p.N, static_cast<int>(p.N * p.ldc_mul));
    if (p.M == 0 || p.N == 0) ldc = 0;

    init_matrix(A, rowsA, colsA, lda, false, 0);
    init_matrix(B, rowsB, colsB, ldb, false, 1);
    
    C_actual.resize(std::max(p.M,1) * ldc);
    std::fill(C_actual.begin(), C_actual.end(), 1.23f); // Pre-fill with a known value
    C_expected = C_actual; // Copy for reference calculation if beta != 0

    // Use alpha=1.0f for reference, as xdnn_sgemm only supports alpha=1.0f
    reference_sgemm_computation(p.transA, p.transB, p.M, p.N, p.K, 1.0f, A, lda, B, ldb, p.beta, C_expected, ldc);
    xdnn_sgemm(p.transA, p.transB, p.M, p.N, p.K, p.alpha, A.data(), lda, B.data(), ldb, p.beta, C_actual.data(), ldc);
    
    verify_matrices(p.M, p.N, ldc);
}

TEST_P(SgemmParamTest, XdnnSgemmSingleThread) {
    SgemmParams p = GetParam();
    int rowsA = p.transA ? p.K : p.M;
    int colsA = p.transA ? p.M : p.K;
    int lda = std::max(colsA, static_cast<int>(colsA * p.lda_mul));
    if (rowsA == 0 || colsA == 0) lda = 0;

    int rowsB = p.transB ? p.N : p.K;
    int colsB = p.transB ? p.K : p.N;
    int ldb = std::max(colsB, static_cast<int>(colsB * p.ldb_mul));
    if (rowsB == 0 || colsB == 0) ldb = 0;
    
    int ldc = std::max(p.N, static_cast<int>(p.N * p.ldc_mul));
    if (p.M == 0 || p.N == 0) ldc = 0;

    init_matrix(A, rowsA, colsA, lda, false, 0);
    init_matrix(B, rowsB, colsB, ldb, false, 1);

    C_actual.resize(std::max(p.M,1) * ldc);
    std::fill(C_actual.begin(), C_actual.end(), 1.23f); 
    C_expected = C_actual;

    reference_sgemm_computation(p.transA, p.transB, p.M, p.N, p.K, 1.0f, A, lda, B, ldb, p.beta, C_expected, ldc);
    xdnn_sgemm_single_thread(p.transA, p.transB, p.M, p.N, p.K, p.alpha, A.data(), lda, B.data(), ldb, p.beta, C_actual.data(), ldc);
    
    verify_matrices(p.M, p.N, ldc);
}

// Test xdnn_sgemm_packb and xdnn_sgemm_compute together
TEST_P(SgemmParamTest, XdnnSgemmPackBAndCompute) {
    SgemmParams p = GetParam();
    // For packb+compute, B is not transposed before packing in this test logic, 
    // the transB flag tells packb how B is currently stored.
    // The compute function always takes A and packedB (which is effectively non-transposed KxN B)

    int rowsA = p.transA ? p.K : p.M;
    int colsA = p.transA ? p.M : p.K;
    int lda = std::max(colsA, static_cast<int>(colsA * p.lda_mul));
    if (rowsA == 0 || colsA == 0) lda = 0;

    // B for packing: if transB=false, B is KxN. if transB=true, B is NxK.
    // packedB will always represent KxN data for the compute kernel.
    int rowsB_orig = p.transB ? p.N : p.K;
    int colsB_orig = p.transB ? p.K : p.N;
    int ldb_orig = std::max(colsB_orig, static_cast<int>(colsB_orig * p.ldb_mul));
    if (rowsB_orig == 0 || colsB_orig == 0) ldb_orig = 0;

    int ldc = std::max(p.N, static_cast<int>(p.N * p.ldc_mul));
    if (p.M == 0 || p.N == 0) ldc = 0;

    init_matrix(A, rowsA, colsA, lda, false, 0);
    std::vector<float> B_original; // B as it is before packing
    init_matrix(B_original, rowsB_orig, colsB_orig, ldb_orig, false, 1);

    // packedB needs to be KxN. Size K*N is a guess, might need specific library function for size.
    if (p.K > 0 && p.N > 0) {
      packedB_actual.resize(p.K * p.N); 
    } else {
      packedB_actual.clear();
    }
    
    if (p.K > 0 && p.N > 0) { // Only pack if there's something to pack
        xdnn_sgemm_packb(p.transB, p.N, p.K, B_original.data(), ldb_orig, packedB_actual.data());
    }

    C_actual.resize(std::max(p.M,1) * ldc);
    std::fill(C_actual.begin(), C_actual.end(), 1.23f);
    C_expected = C_actual;

    // Reference uses original B and its transpose flag
    reference_sgemm_computation(p.transA, p.transB, p.M, p.N, p.K, 1.0f, A, lda, B_original, ldb_orig, p.beta, C_expected, ldc);
    // Compute uses packedB, so effectively transB is false for the A*B part from compute's perspective of B
    xdnn_sgemm_compute(p.transA, p.M, p.N, p.K, p.alpha, A.data(), lda, packedB_actual.data(), p.beta, C_actual.data(), ldc);
    
    verify_matrices(p.M, p.N, ldc);
}

// New tests for sgemm_compute with fused operations

TEST_P(SgemmParamTest, XdnnSgemmComputeSiLu) {
    SgemmParams p = GetParam();
    int rowsA = p.transA ? p.K : p.M;
    int colsA = p.transA ? p.M : p.K;
    int lda = std::max(colsA, static_cast<int>(colsA * p.lda_mul));
    if (rowsA == 0 || colsA == 0) lda = 0;

    int rowsB_orig = p.transB ? p.N : p.K;
    int colsB_orig = p.transB ? p.K : p.N;
    int ldb_orig = std::max(colsB_orig, static_cast<int>(colsB_orig * p.ldb_mul));
    if (rowsB_orig == 0 || colsB_orig == 0) ldb_orig = 0;
    
    int ldc = std::max(p.N, static_cast<int>(p.N * p.ldc_mul));
    if (p.M == 0 || p.N == 0) ldc = 0;

    init_matrix(A, rowsA, colsA, lda, false, 0);
    std::vector<float> B_original;
    init_matrix(B_original, rowsB_orig, colsB_orig, ldb_orig, false, 1);

    if (p.K > 0 && p.N > 0) {
        packedB_actual.resize(p.K * p.N);
        xdnn_sgemm_packb(p.transB, p.N, p.K, B_original.data(), ldb_orig, packedB_actual.data());
    } else {
        packedB_actual.clear();
    }

    C_actual.resize(std::max(p.M, 1) * ldc);
    std::fill(C_actual.begin(), C_actual.end(), 1.23f); // Known pre-fill
    C_expected = C_actual;

    reference_sgemm_computation(p.transA, p.transB, p.M, p.N, p.K, 1.0f, A, lda, B_original, ldb_orig, p.beta, C_expected, ldc);
    apply_silu_to_matrix(C_expected, p.M, p.N, ldc);
    
    xdnn_sgemm_compute_silu(p.transA, p.M, p.N, p.K, p.alpha, A.data(), lda, packedB_actual.data(), p.beta, C_actual.data(), ldc);
    
    verify_matrices(p.M, p.N, ldc);
}

TEST_P(SgemmParamTest, XdnnSgemmComputeGeLu) {
    SgemmParams p = GetParam();
    int rowsA = p.transA ? p.K : p.M;
    int colsA = p.transA ? p.M : p.K;
    int lda = std::max(colsA, static_cast<int>(colsA * p.lda_mul));
    if (rowsA == 0 || colsA == 0) lda = 0;

    int rowsB_orig = p.transB ? p.N : p.K;
    int colsB_orig = p.transB ? p.K : p.N;
    int ldb_orig = std::max(colsB_orig, static_cast<int>(colsB_orig * p.ldb_mul));
    if (rowsB_orig == 0 || colsB_orig == 0) ldb_orig = 0;
    
    int ldc = std::max(p.N, static_cast<int>(p.N * p.ldc_mul));
    if (p.M == 0 || p.N == 0) ldc = 0;

    init_matrix(A, rowsA, colsA, lda, false, 0);
    std::vector<float> B_original;
    init_matrix(B_original, rowsB_orig, colsB_orig, ldb_orig, false, 1);

    if (p.K > 0 && p.N > 0) {
        packedB_actual.resize(p.K * p.N);
        xdnn_sgemm_packb(p.transB, p.N, p.K, B_original.data(), ldb_orig, packedB_actual.data());
    } else {
        packedB_actual.clear();
    }

    C_actual.resize(std::max(p.M, 1) * ldc);
    std::fill(C_actual.begin(), C_actual.end(), 1.23f);
    C_expected = C_actual;

    reference_sgemm_computation(p.transA, p.transB, p.M, p.N, p.K, 1.0f, A, lda, B_original, ldb_orig, p.beta, C_expected, ldc);
    apply_gelu_to_matrix(C_expected, p.M, p.N, ldc);
    
    xdnn_sgemm_compute_gelu(p.transA, p.M, p.N, p.K, p.alpha, A.data(), lda, packedB_actual.data(), p.beta, C_actual.data(), ldc);
    
    verify_matrices(p.M, p.N, ldc);
}

TEST_P(SgemmParamTest, XdnnSgemmComputeBiasAdd) {
    SgemmParams p = GetParam();
    int rowsA = p.transA ? p.K : p.M;
    int colsA = p.transA ? p.M : p.K;
    int lda = std::max(colsA, static_cast<int>(colsA * p.lda_mul));
    if (rowsA == 0 || colsA == 0) lda = 0;

    int rowsB_orig = p.transB ? p.N : p.K;
    int colsB_orig = p.transB ? p.K : p.N;
    int ldb_orig = std::max(colsB_orig, static_cast<int>(colsB_orig * p.ldb_mul));
    if (rowsB_orig == 0 || colsB_orig == 0) ldb_orig = 0;
    
    int ldc = std::max(p.N, static_cast<int>(p.N * p.ldc_mul));
    if (p.M == 0 || p.N == 0) ldc = 0;

    init_matrix(A, rowsA, colsA, lda, false, 0);
    std::vector<float> B_original;
    init_matrix(B_original, rowsB_orig, colsB_orig, ldb_orig, false, 1);

    if (p.K > 0 && p.N > 0) {
        packedB_actual.resize(p.K * p.N);
        xdnn_sgemm_packb(p.transB, p.N, p.K, B_original.data(), ldb_orig, packedB_actual.data());
    } else {
        packedB_actual.clear();
    }
    
    bias.resize(p.N);
    if (p.N > 0) {
        std::mt19937 gen_bias(1337 + 2);
        std::uniform_real_distribution<float> distrib_bias(-1.0f, 1.0f);
        for (int i = 0; i < p.N; ++i) bias[i] = distrib_bias(gen_bias);
    }


    C_actual.resize(std::max(p.M, 1) * ldc);
    std::fill(C_actual.begin(), C_actual.end(), 1.23f);
    C_expected = C_actual;

    reference_sgemm_computation(p.transA, p.transB, p.M, p.N, p.K, 1.0f, A, lda, B_original, ldb_orig, p.beta, C_expected, ldc);
    apply_bias_add_to_matrix(C_expected, p.M, p.N, ldc, bias);
    
    xdnn_sgemm_compute_biasadd(p.transA, p.M, p.N, p.K, p.alpha, A.data(), lda, packedB_actual.data(), p.beta, C_actual.data(), ldc, bias.data());
    
    verify_matrices(p.M, p.N, ldc);
}

TEST_P(SgemmParamTest, XdnnSgemmComputeBiasAddReLu) {
    SgemmParams p = GetParam();
    int rowsA = p.transA ? p.K : p.M;
    int colsA = p.transA ? p.M : p.K;
    int lda = std::max(colsA, static_cast<int>(colsA * p.lda_mul));
    if (rowsA == 0 || colsA == 0) lda = 0;

    int rowsB_orig = p.transB ? p.N : p.K;
    int colsB_orig = p.transB ? p.K : p.N;
    int ldb_orig = std::max(colsB_orig, static_cast<int>(colsB_orig * p.ldb_mul));
    if (rowsB_orig == 0 || colsB_orig == 0) ldb_orig = 0;
    
    int ldc = std::max(p.N, static_cast<int>(p.N * p.ldc_mul));
    if (p.M == 0 || p.N == 0) ldc = 0;

    init_matrix(A, rowsA, colsA, lda, false, 0);
    std::vector<float> B_original;
    init_matrix(B_original, rowsB_orig, colsB_orig, ldb_orig, false, 1);

    if (p.K > 0 && p.N > 0) {
        packedB_actual.resize(p.K * p.N);
        xdnn_sgemm_packb(p.transB, p.N, p.K, B_original.data(), ldb_orig, packedB_actual.data());
    } else {
        packedB_actual.clear();
    }

    bias.resize(p.N);
    if (p.N > 0) {
        std::mt19937 gen_bias(1337 + 2);
        std::uniform_real_distribution<float> distrib_bias(-1.0f, 1.0f);
        for (int i = 0; i < p.N; ++i) bias[i] = distrib_bias(gen_bias);
    }

    C_actual.resize(std::max(p.M, 1) * ldc);
    std::fill(C_actual.begin(), C_actual.end(), 1.23f);
    C_expected = C_actual;

    reference_sgemm_computation(p.transA, p.transB, p.M, p.N, p.K, 1.0f, A, lda, B_original, ldb_orig, p.beta, C_expected, ldc);
    apply_bias_add_to_matrix(C_expected, p.M, p.N, ldc, bias);
    apply_relu_to_matrix(C_expected, p.M, p.N, ldc);
    
    xdnn_sgemm_compute_biasadd_relu(p.transA, p.M, p.N, p.K, p.alpha, A.data(), lda, packedB_actual.data(), p.beta, C_actual.data(), ldc, bias.data());
    
    verify_matrices(p.M, p.N, ldc);
}

TEST_P(SgemmParamTest, XdnnSgemmComputeResidential) {
    SgemmParams p = GetParam();
    int rowsA = p.transA ? p.K : p.M;
    int colsA = p.transA ? p.M : p.K;
    int lda = std::max(colsA, static_cast<int>(colsA * p.lda_mul));
    if (rowsA == 0 || colsA == 0) lda = 0;

    int rowsB_orig = p.transB ? p.N : p.K;
    int colsB_orig = p.transB ? p.K : p.N;
    int ldb_orig = std::max(colsB_orig, static_cast<int>(colsB_orig * p.ldb_mul));
    if (rowsB_orig == 0 || colsB_orig == 0) ldb_orig = 0;
    
    int ldc = std::max(p.N, static_cast<int>(p.N * p.ldc_mul));
    if (p.M == 0 || p.N == 0) ldc = 0;
    int ldres = p.N; // Assuming res is M x N with ldres = N for tests

    init_matrix(A, rowsA, colsA, lda, false, 0);
    std::vector<float> B_original;
    init_matrix(B_original, rowsB_orig, colsB_orig, ldb_orig, false, 1);

    if (p.K > 0 && p.N > 0) {
        packedB_actual.resize(p.K * p.N);
        xdnn_sgemm_packb(p.transB, p.N, p.K, B_original.data(), ldb_orig, packedB_actual.data());
    } else {
        packedB_actual.clear();
    }

    bias.resize(p.N);
    if (p.N > 0) {
        std::mt19937 gen_bias(1337 + 2);
        std::uniform_real_distribution<float> distrib_bias(-1.0f, 1.0f);
        for (int i = 0; i < p.N; ++i) bias[i] = distrib_bias(gen_bias);
    }
    
    res.resize(std::max(p.M,1) * ldres);
    if (p.M > 0 && p.N > 0) init_matrix(res, p.M, p.N, ldres, false, 3);


    C_actual.resize(std::max(p.M, 1) * ldc);
    std::fill(C_actual.begin(), C_actual.end(), 1.23f);
    C_expected = C_actual;

    reference_sgemm_computation(p.transA, p.transB, p.M, p.N, p.K, 1.0f, A, lda, B_original, ldb_orig, p.beta, C_expected, ldc);
    apply_residential_to_matrix(C_expected, p.M, p.N, ldc, bias, res, ldres);
    
    xdnn_sgemm_compute_residential(p.transA, p.M, p.N, p.K, p.alpha, A.data(), lda, packedB_actual.data(), p.beta, C_actual.data(), ldc, bias.data(), res.data(), ldres);
    
    verify_matrices(p.M, p.N, ldc);
}

TEST_P(SgemmResExtParamTest, XdnnSgemmComputeResExt) {
    SgemmResExtParams p = GetParam(); // Use SgemmResExtParams
    int rowsA = p.transA ? p.K : p.M;
    int colsA = p.transA ? p.M : p.K;
    int lda = std::max(colsA, static_cast<int>(colsA * p.lda_mul));
    if (rowsA == 0 || colsA == 0) lda = 0;

    int rowsB_orig = p.transB ? p.N : p.K;
    int colsB_orig = p.transB ? p.K : p.N;
    int ldb_orig = std::max(colsB_orig, static_cast<int>(colsB_orig * p.ldb_mul));
    if (rowsB_orig == 0 || colsB_orig == 0) ldb_orig = 0;
    
    int ldc = std::max(p.N, static_cast<int>(p.N * p.ldc_mul));
    if (p.M == 0 || p.N == 0) ldc = 0;
    int ldres = p.N;

    init_matrix(A, rowsA, colsA, lda, false, 0);
    std::vector<float> B_original;
    init_matrix(B_original, rowsB_orig, colsB_orig, ldb_orig, false, 1);

    if (p.K > 0 && p.N > 0) {
        packedB_actual.resize(p.K * p.N);
        xdnn_sgemm_packb(p.transB, p.N, p.K, B_original.data(), ldb_orig, packedB_actual.data());
    } else {
        packedB_actual.clear();
    }

    bias.resize(p.N);
     if (p.N > 0) {
        std::mt19937 gen_bias(1337 + 2);
        std::uniform_real_distribution<float> distrib_bias(-1.0f, 1.0f);
        for (int i = 0; i < p.N; ++i) bias[i] = distrib_bias(gen_bias);
    }
    
    res.resize(std::max(p.M,1) * ldres);
    if (p.M > 0 && p.N > 0) init_matrix(res, p.M, p.N, ldres, false, 3);

    C_actual.resize(std::max(p.M, 1) * ldc);
    std::fill(C_actual.begin(), C_actual.end(), 1.23f);
    C_expected = C_actual;

    reference_sgemm_computation(p.transA, p.transB, p.M, p.N, p.K, 1.0f, A, lda, B_original, ldb_orig, p.beta, C_expected, ldc);
    apply_resext_to_matrix(C_expected, p.M, p.N, ldc, bias, p.gamma, res, ldres); // Use p.gamma
    
    xdnn_sgemm_compute_resext(p.transA, p.M, p.N, p.K, p.alpha, A.data(), lda, packedB_actual.data(), p.beta, C_actual.data(), ldc, bias.data(), p.gamma, res.data(), ldres);
    
    verify_matrices(p.M, p.N, ldc);
}

TEST_P(SgemmParamTest, XdnnSgemmComputeResMul) {
    SgemmParams p = GetParam();
    int rowsA = p.transA ? p.K : p.M;
    int colsA = p.transA ? p.M : p.K;
    int lda = std::max(colsA, static_cast<int>(colsA * p.lda_mul));
    if (rowsA == 0 || colsA == 0) lda = 0;

    int rowsB_orig = p.transB ? p.N : p.K;
    int colsB_orig = p.transB ? p.K : p.N;
    int ldb_orig = std::max(colsB_orig, static_cast<int>(colsB_orig * p.ldb_mul));
    if (rowsB_orig == 0 || colsB_orig == 0) ldb_orig = 0;
    
    int ldc = std::max(p.N, static_cast<int>(p.N * p.ldc_mul));
    if (p.M == 0 || p.N == 0) ldc = 0;
    int ldres = p.N;

    init_matrix(A, rowsA, colsA, lda, false, 0);
    std::vector<float> B_original;
    init_matrix(B_original, rowsB_orig, colsB_orig, ldb_orig, false, 1);

    if (p.K > 0 && p.N > 0) {
        packedB_actual.resize(p.K * p.N);
        xdnn_sgemm_packb(p.transB, p.N, p.K, B_original.data(), ldb_orig, packedB_actual.data());
    } else {
        packedB_actual.clear();
    }
    
    res.resize(std::max(p.M,1) * ldres);
    if (p.M > 0 && p.N > 0) init_matrix(res, p.M, p.N, ldres, false, 3);


    C_actual.resize(std::max(p.M, 1) * ldc);
    std::fill(C_actual.begin(), C_actual.end(), 1.23f);
    C_expected = C_actual;

    reference_sgemm_computation(p.transA, p.transB, p.M, p.N, p.K, 1.0f, A, lda, B_original, ldb_orig, p.beta, C_expected, ldc);
    apply_resmul_to_matrix(C_expected, p.M, p.N, ldc, res, ldres);
    
    xdnn_sgemm_compute_resmul(p.transA, p.M, p.N, p.K, p.alpha, A.data(), lda, packedB_actual.data(), p.beta, C_actual.data(), ldc, res.data(), ldres);
    
    verify_matrices(p.M, p.N, ldc);
}


// Reference implementation for packb (row-major KxN output)
void reference_packb(bool transB, int N, int K, const float* B, int ldb, float* packedB) {
    // Output is always KxN, row-major
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            if (transB) {
                // B is NxK, so B[n * ldb + k]
                packedB[k * N + n] = B[n * ldb + k];
            } else {
                // B is KxN, so B[k * ldb + n]
                packedB[k * N + n] = B[k * ldb + n];
            }
        }
    }
}

TEST(SgemmPackBTest, PackBCorrectness) {
    // Test a few shapes and both transB cases
    const int N = 5, K = 4;
    for (bool transB : {false, true}) {
        int rowsB = transB ? N : K;
        int colsB = transB ? K : N;
        int ldb = colsB; // tight
        std::vector<float> B(rowsB * ldb);
        // Fill B with sequential values for easy checking
        for (int r = 0; r < rowsB; ++r)
            for (int c = 0; c < colsB; ++c)
                B[r * ldb + c] = 100.0f + r * colsB + c;
        std::vector<float> packed_ref(K * N, -999.0f);
        std::vector<float> packed_func(K * N, -888.0f);
        reference_packb(transB, N, K, B.data(), ldb, packed_ref.data());
        xdnn_sgemm_packb(transB, N, K, B.data(), ldb, packed_func.data());
        for (int i = 0; i < K * N; ++i) {
            ASSERT_FLOAT_EQ(packed_func[i], packed_ref[i]) << "Mismatch at packedB[" << i << "] for transB=" << transB;
        }
    }
}


INSTANTIATE_TEST_SUITE_P(
    SgemmTests, SgemmParamTest,
    ::testing::Values(
        // M, N, K, lda_mul, ldb_mul, ldc_mul, transA, transB, alpha, beta
        SgemmParams{16, 16, 16, 1.0f, 1.0f, 1.0f, false, false, 1.0f, 0.0f}, 
        SgemmParams{32, 32, 32, 1.0f, 1.0f, 1.0f, false, false, 1.0f, 1.0f}, 
        SgemmParams{15, 25, 35, 1.0f, 1.0f, 1.0f, false, false, 1.0f, 0.0f},
        SgemmParams{16, 16, 16, 1.5f, 1.5f, 1.5f, false, false, 1.0f, 0.0f},
        SgemmParams{8, 8, 8, 1.0f, 1.0f, 1.0f, false, false, 1.0f, 0.0f},
        SgemmParams{0, 16, 16, 1.0f, 1.0f, 1.0f, false, false, 1.0f, 1.0f},
        SgemmParams{16, 0, 16, 1.0f, 1.0f, 1.0f, false, false, 1.0f, 1.0f}
    )
);

// Instantiation for the new fused operation tests
INSTANTIATE_TEST_SUITE_P(
    SgemmComputeFusedOpTests, SgemmParamTest,
    ::testing::Values(
        SgemmParams{16, 16, 16, 1.0f, 1.0f, 1.0f, false, false, 1.0f, 0.0f},
        SgemmParams{32, 32, 32, 1.0f, 1.0f, 1.0f, false, false, 1.0f, 1.0f},
        SgemmParams{16, 16, 16, 1.5f, 1.2f, 1.3f, false, false, 1.0f, 0.0f}
    )
);

INSTANTIATE_TEST_SUITE_P(
    SgemmComputeResExtOpTests, SgemmResExtParamTest,
    ::testing::Values(
        SgemmResExtParams{{16, 16, 16, 1.0f, 1.0f, 1.0f, false, false, 1.0f, 0.0f}, 0.5f},
        SgemmResExtParams{{32, 32, 32, 1.0f, 1.0f, 1.0f, false, false, 1.0f, 1.0f}, 1.5f}
    )
);

// Test for small_sgemm correctness
TEST(SmallSgemmTest, BasicCorrectness) {
    // Test a few small shapes
    const std::vector<std::tuple<int, int, int>> shapes = {
        {2, 2, 2}, {3, 3, 3}, {4, 2, 3}, {1, 1, 1}, {0, 2, 2}, {2, 0, 2}, {2, 2, 0}
    };
    for (const auto& tup : shapes) {
        int M, N, K;
        std::tie(M, N, K) = tup;
        int lda = K > 0 ? K : 1;
        int ldb = N > 0 ? N : 1;
        int ldc = N > 0 ? N : 1;
        std::vector<float> A(M * lda);
        std::vector<float> B(K * ldb);
        std::vector<float> C(M * ldc, 1.23f);
        std::vector<float> C_ref = C;
        // Fill A and B with sequential values for determinism
        for (int i = 0; i < (int)A.size(); ++i) A[i] = 1.0f + i;
        for (int i = 0; i < (int)B.size(); ++i) B[i] = 2.0f + i;
        // Reference computation: C = A * B
        reference_sgemm_computation(false, false, M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C_ref, ldc);
        // Test small_sgemm
        small_sgemm(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc);
        // Compare
        ASSERT_EQ(C.size(), C_ref.size());
        for (size_t i = 0; i < C.size(); ++i) {
            ASSERT_NEAR(C[i], C_ref[i], 1e-5f) << "Mismatch at index " << i << " for shape M=" << M << ",N=" << N << ",K=" << K;
        }
    }
}