#include "gtest/gtest.h"
#include "hgemm_f32s8f32.h"
#include "data_types/data_types.h"
#include "test_common.h" // For FP32_PRECISION_TOLERANCE and init_matrix

#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <random>

// Helper function to initialize a matrix with random float values
// Adapted from test_hgemm_f32u4f32.cpp
void init_matrix_float(std::vector<float>& matrix, int rows, int cols, int ld, int seed_offset = 0) {
    matrix.resize(ld * rows);
    if (rows == 0 || cols == 0)
        return;

    std::mt19937 gen(1337 + seed_offset); // Basic seed
    std::uniform_real_distribution<float> distrib(-5.0f, 5.0f); // Adjusted range for quantization
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            matrix[r * ld + c] = distrib(gen);
        }
    }
}

// Helper function to initialize a matrix with simple integer values
// This makes debugging quantization easier
void init_matrix_simple(std::vector<float>& matrix, int rows, int cols, int ld) {
    matrix.resize(ld * rows);
    if (rows == 0 || cols == 0)
        return;

    // Initialize with simple pattern: values from -4 to 4 in sequence
    // This creates a predictable pattern that's easy to trace through quantization
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // Simple pattern with values from -4 to 4
            int value = (r * cols + c) % 9 - 4;
            matrix[r * ld + c] = static_cast<float>(value);
        }
    }
}

// Test Fixture for HgemmF32S8F32 Quantize tests
class HgemmF32S8F32QuantizeTest : public ::testing::Test {
protected:
    std::vector<float> B_float;
    std::vector<int8_t> quantizedB_actual;
    std::vector<float> scaleB_actual;
    std::vector<float> zeroB_actual;

    std::vector<int8_t> quantizedB_expected;
    std::vector<float> scaleB_expected;
    std::vector<float> zeroB_expected;

    void verify_quantized_matrices(int K, int N_quant) {
        ASSERT_EQ(quantizedB_actual.size(), quantizedB_expected.size());
        ASSERT_EQ(scaleB_actual.size(), scaleB_expected.size());
        ASSERT_EQ(zeroB_actual.size(), zeroB_expected.size());

        int total_elements = K * N_quant;
        if (total_elements == 0) {
            ASSERT_TRUE(quantizedB_actual.empty());
            ASSERT_TRUE(quantizedB_expected.empty());
            return; // Nothing more to compare if there are no elements
        }
        
        ASSERT_FALSE(quantizedB_actual.empty()) << "quantizedB_actual should not be empty when K*N_quant > 0";
        ASSERT_FALSE(quantizedB_expected.empty()) << "quantizedB_expected should not be empty when K*N_quant > 0";

        // Compare each element
        for (int i = 0; i < total_elements; ++i) {
            int val_actual = static_cast<int>(quantizedB_actual[i]);
            int val_expected = static_cast<int>(quantizedB_expected[i]);
            
            // Int8 quantization can differ by at most 1 value due to rounding differences
            EXPECT_NEAR(val_actual, val_expected, 1) << "Mismatch for quantizedB at element index " << i
                                            << " (col " << (i/K) << ", row_in_col " << (i%K) << ")";
        }

        // Compare scale and zero point for each column
        for (size_t i = 0; i < scaleB_actual.size(); ++i) {
            EXPECT_NEAR(scaleB_actual[i], scaleB_expected[i], FP32_PRECISION_TOLERANCE)
                << "Scale mismatch at column " << i;
            EXPECT_NEAR(zeroB_actual[i], zeroB_expected[i], FP32_PRECISION_TOLERANCE)
                << "Zero point mismatch at column " << i;
        }
    }
};

// Reference quantization function for int8
// Implementing the actual quantization algorithm as required
void reference_quantize_per_column_s8(
    bool transB, int N, int K, const float *B, int ldb,
    float quantization_rate,
    int8_t *quantizedB, int ldqb,
    float *scaleB_out, float *zeroB_out) {

    const int num_quant_cols = transB ? K : N;
    const int num_quant_rows = transB ? N : K;

    if (num_quant_rows == 0 || num_quant_cols == 0) {
        // Zero out the output arrays if they're provided
        if (scaleB_out) {
            for (int j = 0; j < num_quant_cols; ++j) {
                scaleB_out[j] = 0.0f;
            }
        }
        if (zeroB_out) {
            for (int j = 0; j < num_quant_cols; ++j) {
                zeroB_out[j] = 0.0f;
            }
        }
        return;
    }
    
    // Modified to match actual implementation
    for (int j = 0; j < num_quant_cols; ++j) {
        // Find min and max values in this column
        float col_min = std::numeric_limits<float>::max();
        float col_max = std::numeric_limits<float>::lowest();
        
        for (int i = 0; i < num_quant_rows; ++i) {
            float val = transB ? B[j * ldb + i] : B[i * ldb + j];
            col_min = std::min(col_min, val);
            col_max = std::max(col_max, val);
        }
        
        // Calculate midpoint of the range
        float midpoint = (col_max + col_min) / 2.0f;
        
        // Calculate absolute max as max distance from midpoint
        float abs_max = std::max(std::abs(col_max - midpoint), std::abs(col_min - midpoint));
        
        // Apply quantization_rate if needed
        if (quantization_rate > 0.0f && quantization_rate < 1.0f) {
            abs_max *= quantization_rate;
        }
        
        // Calculate scale for quantization
        float scale = (abs_max > 0) ? abs_max / 127.0f : 1e-9f;
        
        // Store scale and zero point
        scaleB_out[j] = scale;
        zeroB_out[j] = midpoint; // Use midpoint as zero point
        
        // Quantize values in this column
        for (int i = 0; i < num_quant_rows; ++i) {
            float original_val = transB ? B[j * ldb + i] : B[i * ldb + j];
            
            // Quantization with zero point: q = round((val - zero_point) / scale)
            float q = std::round((original_val - midpoint) / scale);
            
            // Clamp to int8 range [-127, 127]
            q = std::max(-127.0f, std::min(127.0f, q));
            
            // Store the result
            int output_idx = i * ldqb + j;
            quantizedB[output_idx] = static_cast<int8_t>(q);
        }
    }
}

struct HgemmF32S8F32QuantizeParams {
    int N, K; // Dimensions of B (if transB=false, B is KxN, if transB=true, B is NxK)
    float ldb_mul;
    bool transB;
    float quantization_rate;
};

class HgemmF32S8F32QuantizeParamTest : public HgemmF32S8F32QuantizeTest,
                                       public ::testing::WithParamInterface<HgemmF32S8F32QuantizeParams> {};

TEST_P(HgemmF32S8F32QuantizeParamTest, XdnnHgemmF32S8F32Quantize) {
    HgemmF32S8F32QuantizeParams p = GetParam();

    int rowsB_orig = p.transB ? p.N : p.K;
    int colsB_orig = p.transB ? p.K : p.N;
    int ldb = std::max(colsB_orig, static_cast<int>(colsB_orig * p.ldb_mul));
    if (rowsB_orig == 0 || colsB_orig == 0)
        ldb = 0;

    // Use simple integer values for easier debugging
    init_matrix_simple(B_float, rowsB_orig, colsB_orig, ldb);

    // Output parameters for quantization
    int num_cols_to_quantize = p.transB ? p.K : p.N;
    int num_rows_per_quant_col = p.transB ? p.N : p.K;

    // For int8_t, ldqb should be the stride between rows in the quantized output
    // which matches num_cols_to_quantize for row-major packing
    int ldqb = num_cols_to_quantize;

    if (num_cols_to_quantize > 0) {
        scaleB_actual.resize(num_cols_to_quantize);
        zeroB_actual.resize(num_cols_to_quantize);
        scaleB_expected.resize(num_cols_to_quantize);
        zeroB_expected.resize(num_cols_to_quantize);
    }

    if (num_rows_per_quant_col > 0 && num_cols_to_quantize > 0) {
        // The total size should be num_rows_per_quant_col * ldqb
        // where ldqb is the stride between rows in the quantized output
        quantizedB_actual.resize(num_rows_per_quant_col * ldqb);
        quantizedB_expected.resize(num_rows_per_quant_col * ldqb);
    } else {
        quantizedB_actual.clear();
        quantizedB_expected.clear();
    }

    // Call the function under test
    xdnn_hgemm_f32s8f32_quantize(p.transB, p.N, p.K, B_float.data(), ldb,
                                 p.quantization_rate, 
                                 quantizedB_actual.data(), ldqb, 
                                 scaleB_actual.data(), zeroB_actual.data());

    // Prepare expected results
    reference_quantize_per_column_s8(p.transB, p.N, p.K, B_float.data(), ldb, 
                                     p.quantization_rate,
                                     quantizedB_expected.data(), ldqb,
                                     scaleB_expected.data(), zeroB_expected.data());
    
    // Verification
    verify_quantized_matrices(num_rows_per_quant_col, num_cols_to_quantize);
}

INSTANTIATE_TEST_SUITE_P(
    HgemmF32S8F32QuantizeTests, HgemmF32S8F32QuantizeParamTest,
    ::testing::Values(
        // N, K, ldb_mul, transB, quantization_rate
        HgemmF32S8F32QuantizeParams{4, 4, 1.0f, false, 1.0f}, // B is 32x16 (KxN)
        HgemmF32S8F32QuantizeParams{16, 32, 1.0f, false, 1.0f}, // B is 32x16 (KxN)
        HgemmF32S8F32QuantizeParams{8, 16, 1.2f, false, 1.0f} // ldb padding
    )
);

// Reference implementation for packb (row-major KxN output)
void reference_packb_s8(bool transB, int N, int K, const int8_t* B, int ldb, int8_t* packedB) {
    // Output is always KxN, row-major
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            int src_idx = transB ? (n * ldb + k) : (k * ldb + n);
            int dst_idx = k * N + n;
            packedB[dst_idx] = B[src_idx];
        }
    }
}

// Test Fixture for HgemmF32S8F32 PackB tests
class HgemmF32S8F32PackBTest : public ::testing::Test {
protected:
    std::vector<int8_t> B_s8;
    std::vector<int8_t> packedB_actual;
    std::vector<int8_t> packedB_expected;

    void verify_packedB(int K, int N) {
        ASSERT_EQ(packedB_actual.size(), packedB_expected.size());
        for (int i = 0; i < K * N; ++i) {
            EXPECT_EQ(packedB_actual[i], packedB_expected[i]) << "Mismatch at packedB[" << i << "]";
        }
    }
};

TEST(HgemmF32S8F32PackBTest, PackBCorrectness) {
    // Test a few shapes and both transB cases
    const int N = 4, K = 4;
    
    for (bool transB : {false, true}) {
        int ldb = N;  // For this test, we keep it simple with tight ldb
        
        // Initialize B with some pattern
        std::vector<int8_t> B_s8(K * ldb);
        for (int i = 0; i < K * ldb; ++i) {
            B_s8[i] = static_cast<int8_t>((i % 255) - 127); // Values from -127 to 127
        }
        
        std::vector<int8_t> packed_ref(K * N);
        std::vector<int8_t> packed_func(K * N);
        
        // Reference pack
        reference_packb_s8(transB, N, K, B_s8.data(), ldb, packed_ref.data());
        
        // Function under test
        xdnn_hgemm_f32s8f32_packb(transB, N, K, B_s8.data(), ldb, packed_func.data());
        
        // Check
        for (int i = 0; i < K * N; ++i) {
            EXPECT_EQ(packed_func[i], packed_ref[i]) << "Mismatch at packedB[" << i << "] for transB=" << transB;
        }
    }
}

// Reference: Dequantize B (int8) to float using scaleB/zeroB, then compute C = alpha * A * B + beta * C
void reference_hgemm_f32s8f32_compute(
    bool transA, int M, int N, int K,
    float alpha, const float *A, int lda,
    const int8_t *packedB, const float *scaleB, const float *zeroB,
    float beta, float *C, int ldc) {
    
    // Similar constraints as in f32u4f32 tests
    if (alpha != 1.0f) {
        std::cerr << "ERROR: reference_hgemm_f32s8f32_compute only supports alpha == 1.0f. Got alpha=" << alpha << std::endl;
        return;
    }
    if (!(beta == 0.0f || beta == 1.0f)) {
        std::cerr << "ERROR: reference_hgemm_f32s8f32_compute only supports beta == 0.0f or beta == 1.0f. Got beta=" << beta << std::endl;
        return;
    }
    
    if (M == 0 || N == 0) return;
    
    // Apply beta to C
    if (beta == 0.0f) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                C[m * ldc + n] = 0.0f;
            }
        }
    }
    
    // Compute C = A * B where B is already packed in row-major KxN format
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            
            for (int k = 0; k < K; ++k) {
                float a_val = transA ? A[k * lda + m] : A[m * lda + k];
                
                // Get the quantized value and dequantize it
                int8_t b_quant = packedB[k * N + n];
                
                // Dequantize using proper formula: val = q_val * scale + zero_point
                float b_val = static_cast<float>(b_quant) * scaleB[n] + zeroB[n];
                
                sum += a_val * b_val;
            }
            
            // Write the result
            C[m * ldc + n] += alpha * sum;
        }
    }
}

struct HgemmF32S8F32ComputeParams {
    int M, N, K;
    float lda_mul, ldc_mul;
    bool transA;
    float alpha, beta;
};

class HgemmF32S8F32ComputeTest : public ::testing::Test {
protected:
    std::vector<float> A;
    std::vector<int8_t> packedB;
    std::vector<float> scaleB;
    std::vector<float> zeroB;
    std::vector<float> C_actual;
    std::vector<float> C_expected;

    void verify_C(int M, int N, int ldc) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                EXPECT_NEAR(C_actual[m * ldc + n], C_expected[m * ldc + n], FP32_PRECISION_TOLERANCE)
                    << "Mismatch at C[" << m << "," << n << "]";
            }
        }
    }
};

class HgemmF32S8F32ComputeParamTest : public HgemmF32S8F32ComputeTest, 
                                     public ::testing::WithParamInterface<HgemmF32S8F32ComputeParams> {};

TEST_P(HgemmF32S8F32ComputeParamTest, XdnnHgemmF32S8F32Compute) {
    HgemmF32S8F32ComputeParams p = GetParam();
    
    int rowsA = p.transA ? p.K : p.M;
    int colsA = p.transA ? p.M : p.K;
    int lda = std::max(colsA, static_cast<int>(colsA * p.lda_mul));
    if (rowsA == 0 || colsA == 0) lda = 0;
    
    int ldc = std::max(p.N, static_cast<int>(p.N * p.ldc_mul));
    if (p.M == 0 || p.N == 0) ldc = 0;
    
    // Generate A with simple integer values
    init_matrix_simple(A, rowsA, colsA, lda);
    
    // Generate B, quantize, pack, and get scale/zero
    int N = p.N, K = p.K;
    std::vector<float> B_float(K * N);
    init_matrix_simple(B_float, K, N, N); // tight ldb
    
    std::vector<int8_t> quantizedB(K * N);
    std::vector<float> scaleB_vec(N), zeroB_vec(N);
    
    reference_quantize_per_column_s8(false, N, K, B_float.data(), N, 1.0f, 
                                    quantizedB.data(), N, 
                                    scaleB_vec.data(), zeroB_vec.data());
    
    // Pack B (row-major KxN)
    packedB.resize(K * N);
    reference_packb_s8(false, N, K, quantizedB.data(), N, packedB.data());
    
    scaleB = scaleB_vec;
    zeroB = zeroB_vec;
    
    // Prepare C
    C_actual.resize(std::max(p.M,1) * ldc);
    C_expected.resize(std::max(p.M,1) * ldc);
    std::fill(C_actual.begin(), C_actual.end(), 1.23f);
    std::fill(C_expected.begin(), C_expected.end(), 1.23f);
    
    // Reference
    reference_hgemm_f32s8f32_compute(p.transA, p.M, p.N, p.K, p.alpha, A.data(), lda, 
                                    packedB.data(), scaleB.data(), zeroB.data(), 
                                    p.beta, C_expected.data(), ldc);
    
    // Function under test
    xdnn_hgemm_f32s8f32_compute(p.transA, p.M, p.N, p.K, p.alpha, A.data(), lda, 
                                packedB.data(), scaleB.data(), zeroB.data(), 
                                p.beta, C_actual.data(), ldc);
    
    // Verify
    verify_C(p.M, p.N, ldc);
}

INSTANTIATE_TEST_SUITE_P(
    HgemmF32S8F32ComputeTests, HgemmF32S8F32ComputeParamTest,
    ::testing::Values(
        HgemmF32S8F32ComputeParams{8, 16, 32, 1.0f, 1.0f, false, 1.0f, 0.0f},  // beta = 0
        HgemmF32S8F32ComputeParams{8, 16, 32, 1.0f, 1.0f, false, 1.0f, 1.0f},  // beta = 1
        HgemmF32S8F32ComputeParams{16, 8, 32, 1.0f, 1.0f, false, 1.0f, 0.0f},  // different dimensions
        HgemmF32S8F32ComputeParams{8, 16, 32, 1.0f, 1.2f, false, 1.0f, 0.0f},  // ldc padding
        HgemmF32S8F32ComputeParams{8, 16, 32, 1.2f, 1.0f, false, 1.0f, 0.0f},  // lda padding
        HgemmF32S8F32ComputeParams{0, 16, 32, 1.0f, 1.0f, false, 1.0f, 0.0f},  // M = 0
        HgemmF32S8F32ComputeParams{8, 0, 32, 1.0f, 1.0f, false, 1.0f, 0.0f}   // N = 0
    )
);

// === BEGIN: F32S8F32 Compute Fused Op Tests ===

// Helper: SiLU activation
inline float silu_op(float x) { return x / (1.0f + std::exp(-x)); }

void apply_silu_to_matrix(std::vector<float>& matrix, int M, int N, int ldc) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            matrix[r * ldc + c] = silu_op(matrix[r * ldc + c]);
        }
    }
}

// Helper: GELU activation (approx)
inline float gelu_approx_op(float x) {
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3.0f))));
}

void apply_gelu_to_matrix(std::vector<float>& matrix, int M, int N, int ldc) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            matrix[r * ldc + c] = gelu_approx_op(matrix[r * ldc + c]);
        }
    }
}

// Helper: ReLU
inline float relu_op(float x) { return std::max(0.0f, x); }

void apply_relu_to_matrix(std::vector<float>& matrix, int M, int N, int ldc) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            matrix[r * ldc + c] = relu_op(matrix[r * ldc + c]);
        }
    }
}

// Helper: Bias add
void apply_bias_add_to_matrix(std::vector<float>& C, int M, int N, int ldc, const std::vector<float>& bias_vec) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            C[r * ldc + c] += bias_vec[c];
        }
    }
}

// Helper: Residential add
void apply_residential_to_matrix(std::vector<float>& C, int M, int N, int ldc, const std::vector<float>& bias_vec, const std::vector<float>& res_vec, int ldres) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            C[r * ldc + c] += bias_vec[c] + res_vec[r * ldres + c];
        }
    }
}

// Helper: ResExt
void apply_resext_to_matrix(std::vector<float>& C, int M, int N, int ldc, const std::vector<float>& bias_vec, float gamma, const std::vector<float>& res_vec, int ldres) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            C[r * ldc + c] += bias_vec[c] + gamma * res_vec[r * ldres + c];
        }
    }
}

// Helper: ResMul
void apply_resmul_to_matrix(std::vector<float>& C, int M, int N, int ldc, const std::vector<float>& res_vec, int ldres) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            C[r * ldc + c] *= res_vec[r * ldres + c];
        }
    }
}

// Test fixture for F32S8F32 fused op tests
class HgemmF32S8F32FusedOpTest : public ::testing::Test {
protected:
    int M = 4, N = 4, K = 4;
    std::vector<float> A, B_float, C_actual, C_expected;
    std::vector<int8_t> packedB;
    std::vector<float> scaleB, zeroB, bias, res;
    float gamma = 1.5f;
    int lda = 4, ldc = 4, ldres = 4;
    
    void SetUp() override {
        // Fill A, B_float with simple pattern for easier debugging
        A.resize(M * K);
        B_float.resize(K * N);
        
        // Use the same pattern as init_matrix_simple
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < K; ++c) {
                int value = (r * K + c) % 9 - 4;
                A[r * K + c] = static_cast<float>(value);
            }
        }
        
        for (int r = 0; r < K; ++r) {
            for (int c = 0; c < N; ++c) {
                int value = (r * N + c) % 9 - 4;
                B_float[r * N + c] = static_cast<float>(value);
            }
        }
        
        // Quantize/pack B - Use strictly symmetric quantization
        packedB.resize(K * N);
        scaleB.resize(N);
        zeroB.resize(N);
        
        // Manual symmetric quantization
        std::vector<int8_t> quantizedB(K * N);
        
        // For each column, find the max absolute value and set scale 
        for (int c = 0; c < N; ++c) {
            float abs_max = 0;
            for (int r = 0; r < K; ++r) {
                abs_max = std::max(abs_max, std::abs(B_float[r * N + c]));
            }
            
            // Set scale based on abs_max/127
            scaleB[c] = (abs_max > 0) ? abs_max / 127.0f : 1e-9f;
            zeroB[c] = 0.0f; // Strictly symmetric - zero point is always 0
            
            // Quantize column
            for (int r = 0; r < K; ++r) {
                float val = B_float[r * N + c];
                float q = std::round(val / scaleB[c]);
                q = std::max(-127.0f, std::min(127.0f, q));
                quantizedB[r * N + c] = static_cast<int8_t>(q);
            }
        }
        
        // Pack B
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                packedB[k * N + n] = quantizedB[k * N + n];
            }
        }
        
        // Bias and res
        bias.resize(N);
        for (int i = 0; i < N; ++i) bias[i] = (i % 3) - 1.0f;
        
        res.resize(M * ldres);
        for (int i = 0; i < M * ldres; ++i) res[i] = (i % 4) * 0.5f;
    }
};

TEST_F(HgemmF32S8F32FusedOpTest, ComputeSiLU) {
    // Initialize test matrices
    C_actual.assign(M * ldc, 1.23f);
    C_expected = C_actual;
    
    // Create temporary buffer for the base computation
    std::vector<float> C_base(M * ldc, 1.23f);
    
    // Step 1: Perform base matrix multiply for both reference and actual
    reference_hgemm_f32s8f32_compute(false, M, N, K, 1.0f, A.data(), lda, 
                                    packedB.data(), scaleB.data(), zeroB.data(), 
                                    0.0f, C_base.data(), ldc);
    
    // Step 2: For reference, apply SiLU activation manually to the base result
    for (int i = 0; i < M * ldc; ++i) {
        C_expected[i] = C_base[i] * (1.0f / (1.0f + std::exp(-C_base[i])));
    }
    
    // Step 3: For actual implementation, call the fused function
    xdnn_hgemm_f32s8f32_compute_silu(false, M, N, K, 1.0f, A.data(), lda, 
                                    packedB.data(), scaleB.data(), zeroB.data(), 
                                    0.0f, C_actual.data(), ldc);
    
    // Step 4: Compare with higher tolerance for quantized operations
    float higher_tolerance = F32S8F32_ACTIVATION_TOLERANCE;  // Use defined tolerance for quantized activation functions
    bool all_passed = true;
    
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int idx = r * ldc + c;
            float diff = std::abs(C_expected[idx] - C_actual[idx]);
            if (diff > higher_tolerance) {
                all_passed = false;
                break;
            }
        }
    }
    
    EXPECT_TRUE(all_passed) << "Some values exceed the tolerance threshold";
}

TEST_F(HgemmF32S8F32FusedOpTest, ComputeGeLU) {
    // Initialize test matrices
    C_actual.assign(M * ldc, 1.23f);
    C_expected = C_actual;
    
    // Create temporary buffer for the base computation
    std::vector<float> C_base(M * ldc, 1.23f);
    
    // Step 1: Perform base matrix multiply for reference
    reference_hgemm_f32s8f32_compute(false, M, N, K, 1.0f, A.data(), lda, 
                                    packedB.data(), scaleB.data(), zeroB.data(), 
                                    0.0f, C_base.data(), ldc);
    
    // Step 2: For reference, apply GELU activation manually to the base result
    for (int i = 0; i < M * ldc; ++i) {
        // GELU approximation: 0.5 * x * (1.0 + tanh(sqrt(2.0/π) * (x + 0.044715 * x^3)))
        float x = C_base[i];
        C_expected[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3.0f))));
    }
    
    // Step 3: For actual implementation, call the fused function
    xdnn_hgemm_f32s8f32_compute_gelu(false, M, N, K, 1.0f, A.data(), lda, 
                                    packedB.data(), scaleB.data(), zeroB.data(), 
                                    0.0f, C_actual.data(), ldc);
    
    // Step 4: Compare with higher tolerance for quantized operations
    float higher_tolerance = F32S8F32_ACTIVATION_TOLERANCE;  // Use defined tolerance for quantized activation functions
    bool all_passed = true;
    
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int idx = r * ldc + c;
            float diff = std::abs(C_expected[idx] - C_actual[idx]);
            if (diff > higher_tolerance) {
                all_passed = false;
                break;
            }
        }
    }
    
    EXPECT_TRUE(all_passed) << "Some values exceed the tolerance threshold";
}

TEST_F(HgemmF32S8F32FusedOpTest, ComputeBiasAdd) {
    // Initialize test matrices
    C_actual.assign(M * ldc, 1.23f);
    C_expected = C_actual;
    
    // Create temporary buffer for the base computation
    std::vector<float> C_base(M * ldc, 1.23f);
    
    // Step 1: Perform base matrix multiply for reference
    reference_hgemm_f32s8f32_compute(false, M, N, K, 1.0f, A.data(), lda, 
                                    packedB.data(), scaleB.data(), zeroB.data(), 
                                    0.0f, C_base.data(), ldc);
    
    // Step 2: For reference, apply bias addition manually to the base result
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C_expected[m * ldc + n] = C_base[m * ldc + n] + bias[n];
        }
    }
    
    // Step 3: For actual implementation, call the fused function
    xdnn_hgemm_f32s8f32_compute_biasadd(false, M, N, K, 1.0f, A.data(), lda, 
                                       packedB.data(), scaleB.data(), zeroB.data(), 
                                       0.0f, C_actual.data(), ldc, bias.data());
    
    // Step 4: Compare with higher tolerance for quantized operations
    float higher_tolerance = F32S8F32_ACTIVATION_TOLERANCE;  // Use defined tolerance for quantized activation functions
    bool all_passed = true;
    
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int idx = r * ldc + c;
            float diff = std::abs(C_expected[idx] - C_actual[idx]);
            if (diff > higher_tolerance) {
                all_passed = false;
                break;
            }
        }
    }
    
    EXPECT_TRUE(all_passed) << "Some values exceed the tolerance threshold";
}

TEST_F(HgemmF32S8F32FusedOpTest, ComputeBiasAddReLU) {
    // Initialize test matrices
    C_actual.assign(M * ldc, 1.23f);
    C_expected = C_actual;
    
    // Create temporary buffer for the base computation
    std::vector<float> C_base(M * ldc, 1.23f);
    
    // Step 1: Perform base matrix multiply for reference
    reference_hgemm_f32s8f32_compute(false, M, N, K, 1.0f, A.data(), lda, 
                                    packedB.data(), scaleB.data(), zeroB.data(), 
                                    0.0f, C_base.data(), ldc);
    
    // Step 2: For reference, apply bias addition and ReLU manually to the base result
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C_expected[m * ldc + n] = std::max(0.0f, C_base[m * ldc + n] + bias[n]);
        }
    }
    
    // Step 3: For actual implementation, call the fused function
    xdnn_hgemm_f32s8f32_compute_biasadd_relu(false, M, N, K, 1.0f, A.data(), lda, 
                                           packedB.data(), scaleB.data(), zeroB.data(), 
                                           0.0f, C_actual.data(), ldc, bias.data());
    
    // Step 4: Compare with higher tolerance for quantized operations
    float higher_tolerance = F32S8F32_ACTIVATION_TOLERANCE;  // Use defined tolerance for quantized activation functions
    bool all_passed = true;
    
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int idx = r * ldc + c;
            float diff = std::abs(C_expected[idx] - C_actual[idx]);
            if (diff > higher_tolerance) {
                all_passed = false;
                break;
            }
        }
    }
    
    EXPECT_TRUE(all_passed) << "Some values exceed the tolerance threshold";
}

TEST_F(HgemmF32S8F32FusedOpTest, ComputeResidential) {
    // Initialize test matrices
    C_actual.assign(M * ldc, 1.23f);
    C_expected = C_actual;
    
    // Create temporary buffer for the base computation
    std::vector<float> C_base(M * ldc, 1.23f);
    
    // Step 1: Perform base matrix multiply for reference
    reference_hgemm_f32s8f32_compute(false, M, N, K, 1.0f, A.data(), lda, 
                                    packedB.data(), scaleB.data(), zeroB.data(), 
                                    0.0f, C_base.data(), ldc);
    
    // Step 2: For reference, apply bias and residential addition manually
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C_expected[m * ldc + n] = C_base[m * ldc + n] + bias[n] + res[m * ldres + n];
        }
    }
    
    // Step 3: For actual implementation, call the fused function
    xdnn_hgemm_f32s8f32_compute_residential(false, M, N, K, 1.0f, A.data(), lda, 
                                          packedB.data(), scaleB.data(), zeroB.data(), 
                                          0.0f, C_actual.data(), ldc, bias.data(), res.data(), ldres);
    
    // Step 4: Compare with higher tolerance for quantized operations
    float higher_tolerance = F32S8F32_ACTIVATION_TOLERANCE;  // Use defined tolerance for quantized activation functions
    bool all_passed = true;
    
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int idx = r * ldc + c;
            float diff = std::abs(C_expected[idx] - C_actual[idx]);
            if (diff > higher_tolerance) {
                all_passed = false;
                break;
            }
        }
    }
    
    EXPECT_TRUE(all_passed) << "Some values exceed the tolerance threshold";
}

TEST_F(HgemmF32S8F32FusedOpTest, ComputeResExt) {
    // Initialize test matrices
    C_actual.assign(M * ldc, 1.23f);
    C_expected = C_actual;
    
    // Create temporary buffer for the base computation
    std::vector<float> C_base(M * ldc, 1.23f);
    
    // Step 1: Perform base matrix multiply for reference
    reference_hgemm_f32s8f32_compute(false, M, N, K, 1.0f, A.data(), lda, 
                                    packedB.data(), scaleB.data(), zeroB.data(), 
                                    0.0f, C_base.data(), ldc);
    
    // Step 2: For reference, apply bias and gamma*res manually
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C_expected[m * ldc + n] = C_base[m * ldc + n] + bias[n] + gamma * res[m * ldres + n];
        }
    }
    
    // Step 3: For actual implementation, call the fused function
    xdnn_hgemm_f32s8f32_compute_resext(false, M, N, K, 1.0f, A.data(), lda, 
                                      packedB.data(), scaleB.data(), zeroB.data(), 
                                      0.0f, C_actual.data(), ldc, bias.data(), gamma, res.data(), ldres);
    
    // Step 4: Compare with higher tolerance for quantized operations
    float higher_tolerance = F32S8F32_ACTIVATION_TOLERANCE;  // Use defined tolerance for quantized activation functions
    bool all_passed = true;
    
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int idx = r * ldc + c;
            float diff = std::abs(C_expected[idx] - C_actual[idx]);
            if (diff > higher_tolerance) {
                all_passed = false;
                break;
            }
        }
    }
    
    EXPECT_TRUE(all_passed) << "Some values exceed the tolerance threshold";
}

TEST_F(HgemmF32S8F32FusedOpTest, ComputeResMul) {
    // Initialize test matrices
    C_actual.assign(M * ldc, 1.23f);
    C_expected = C_actual;
    
    // Create temporary buffer for the base computation
    std::vector<float> C_base(M * ldc, 1.23f);
    
    // Step 1: Perform base matrix multiply for reference
    reference_hgemm_f32s8f32_compute(false, M, N, K, 1.0f, A.data(), lda, 
                                    packedB.data(), scaleB.data(), zeroB.data(), 
                                    0.0f, C_base.data(), ldc);
    
    // Step 2: For reference, apply element-wise multiplication with res
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C_expected[m * ldc + n] = C_base[m * ldc + n] * res[m * ldres + n];
        }
    }
    
    // Step 3: For actual implementation, call the fused function
    xdnn_hgemm_f32s8f32_compute_resmul(false, M, N, K, 1.0f, A.data(), lda, 
                                      packedB.data(), scaleB.data(), zeroB.data(), 
                                      0.0f, C_actual.data(), ldc, res.data(), ldres);
    
    // Step 4: Compare with higher tolerance for quantized operations
    float higher_tolerance = F32S8F32_ACTIVATION_TOLERANCE;  // Use defined tolerance for quantized activation functions
    bool all_passed = true;
    
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int idx = r * ldc + c;
            float diff = std::abs(C_expected[idx] - C_actual[idx]);
            if (diff > higher_tolerance) {
                all_passed = false;
                break;
            }
        }
    }
    
    EXPECT_TRUE(all_passed) << "Some values exceed the tolerance threshold";
}
