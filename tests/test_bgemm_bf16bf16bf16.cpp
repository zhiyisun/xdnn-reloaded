#include "bgemm_bf16bf16bf16.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

// Helper: Convert XDNN_BF16 to float
inline float bf16_to_float(XDNN_BF16 val) {
    union { uint32_t u; float f; } u = { static_cast<uint32_t>(val) << 16 };
    return u.f;
}
// Helper: Convert float to XDNN_BF16 (truncate lower 16 bits)
inline XDNN_BF16 float_to_bf16(float val) {
    union { float f; uint32_t u; } u = { val };
    return static_cast<XDNN_BF16>(u.u >> 16);
}

// Helper to fill XDNN_BF16 matrix with a pattern
void fill_bf16_matrix(std::vector<XDNN_BF16>& mat, int rows, int cols, int ld, float start = 1.0f) {
    mat.resize(ld * rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            mat[r * ld + c] = float_to_bf16(start + r * cols + c);
}

// Reference: packb (row-major KxN output)
void reference_packb(bool transB, int N, int K, const XDNN_BF16* B, int ldb, XDNN_BF16* packedB, int block_rows, int block_cols) {
    (void)block_rows; (void)block_cols; // unused in reference
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            int src_idx = transB ? (n * ldb + k) : (k * ldb + n);
            int dst_idx = k * N + n;
            packedB[dst_idx] = B[src_idx];
        }
    }
}

// Reference: C = alpha * A * B + beta * C (B is KxN row-major, A is MxK row-major)
void reference_compute(bool transA, int M, int N, int K, float alpha, const XDNN_BF16* A, int lda, const XDNN_BF16* packedB, float beta, XDNN_BF16* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = bf16_to_float(transA ? A[k * lda + m] : A[m * lda + k]);
                float b_val = bf16_to_float(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            float c_val = bf16_to_float(C[m * ldc + n]);
            float result = (beta == 0.0f) ? (alpha * sum) : (alpha * sum + beta * c_val);
            C[m * ldc + n] = float_to_bf16(result);
        }
    }
}

// Reference: SiLU activation
float silu(float x) { return x / (1.0f + std::exp(-x)); }
// Reference: GELU activation
float gelu(float x) { return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3.0f)))); }
// Reference: ReLU
float relu(float x) { return std::max(0.0f, x); }

// Parameterized test for packb
struct PackBParams { int N, K, ldb; bool transB; };
class bgemmBF16BF16BF16PackBTest : public ::testing::TestWithParam<PackBParams> {};
TEST_P(bgemmBF16BF16BF16PackBTest, PackBCorrectness) {
    auto p = GetParam();
    std::vector<XDNN_BF16> B(p.ldb * (p.transB ? p.N : p.K));
    std::vector<XDNN_BF16> packedB(p.K * p.N, 0), packedB_ref(p.K * p.N, 0);
    fill_bf16_matrix(B, p.transB ? p.N : p.K, p.transB ? p.K : p.N, p.ldb);
    reference_packb(p.transB, p.N, p.K, B.data(), p.ldb, packedB_ref.data(), p.K, p.N);
    xdnn_bgemm_bf16bf16bf16_packb(p.transB, p.N, p.K, B.data(), p.ldb, packedB.data(), p.K, p.N);
    for (int i = 0; i < p.K * p.N; ++i) EXPECT_EQ(packedB[i], packedB_ref[i]);
}
INSTANTIATE_TEST_SUITE_P(PackB, bgemmBF16BF16BF16PackBTest, ::testing::Values(
    PackBParams{4, 4, 4, false}, 
    PackBParams{4, 4, 4, true}, 
    PackBParams{8, 8, 10, false}
));

// Parameterized test for compute
struct ComputeParams { int M, N, K; float alpha, beta; bool transA; };
class bgemmBF16BF16BF16ComputeTest : public ::testing::TestWithParam<ComputeParams> {};
TEST_P(bgemmBF16BF16BF16ComputeTest, ComputeCorrectness) {
    auto p = GetParam();
    std::vector<XDNN_BF16> A(p.M * p.K), C(p.M * p.N, static_cast<XDNN_BF16>(1.23f)), C_ref(p.M * p.N, static_cast<XDNN_BF16>(1.23f));
    std::vector<XDNN_BF16> packedB(p.K * p.N);
    fill_bf16_matrix(A, p.M, p.K, p.K);
    fill_bf16_matrix(packedB, p.K, p.N, p.N);
    reference_compute(p.transA, p.M, p.N, p.K, p.alpha, A.data(), p.K, packedB.data(), p.beta, C_ref.data(), p.N);
    xdnn_bgemm_bf16bf16bf16_compute(p.transA, p.M, p.N, p.K, p.alpha, A.data(), p.K, packedB.data(), p.beta, C.data(), p.N);
    for (int i = 0; i < p.M * p.N; ++i) EXPECT_NEAR(static_cast<float>(C[i]), static_cast<float>(C_ref[i]), 1e-3);
}
INSTANTIATE_TEST_SUITE_P(Compute, bgemmBF16BF16BF16ComputeTest, ::testing::Values(
    ComputeParams{2, 2, 2, 1.0f, 0.0f, false}, 
    ComputeParams{2, 2, 2, 1.0f, 1.0f, false},
    ComputeParams{4, 4, 4, 1.0f, 0.0f, false}, 
    ComputeParams{4, 4, 4, 1.0f, 0.0f, false}
));

// Fused op tests (SiLU, GELU, BiasAdd, BiasAddReLU, Residential, ResExt, ResMul)
class bgemmBF16BF16BF16FusedTest : public ::testing::Test {
protected:
    int M = 16, N = 16, K = 16;
    int ldc = 16; // Use N for now, but can be larger for padding tests
    std::vector<XDNN_BF16> A, C, C_ref, bias, res;
    std::vector<XDNN_BF16> packedB;
    float gamma = 1.5f;
    void SetUp() override {
        A.resize(M * K); fill_bf16_matrix(A, M, K, K);
        packedB.resize(K * N); fill_bf16_matrix(packedB, K, N, N);
        C.assign(M * ldc, static_cast<XDNN_BF16>(1.23f)); C_ref = C;
        bias.assign(N, static_cast<XDNN_BF16>(2.0f));
        res.assign(M * ldc, static_cast<XDNN_BF16>(3.0f));
    }
};
TEST_F(bgemmBF16BF16BF16FusedTest, ComputeSiLU) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] = float_to_bf16(silu(bf16_to_float(C_ref[m * ldc + n])));
    xdnn_bgemm_bf16bf16bf16_compute_silu(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(bf16_to_float(C[m * ldc + n]), bf16_to_float(C_ref[m * ldc + n]), 1e-3);
}
TEST_F(bgemmBF16BF16BF16FusedTest, ComputeGELU) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] = float_to_bf16(gelu(bf16_to_float(C_ref[m * ldc + n])));
    xdnn_bgemm_bf16bf16bf16_compute_gelu(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(bf16_to_float(C[m * ldc + n]), bf16_to_float(C_ref[m * ldc + n]), 1e-3);
}
TEST_F(bgemmBF16BF16BF16FusedTest, ComputeBiasAdd) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] = float_to_bf16(bf16_to_float(C_ref[m * ldc + n]) + bf16_to_float(bias[n]));
    xdnn_bgemm_bf16bf16bf16_compute_biasadd(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc, bias.data());
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(bf16_to_float(C[m * ldc + n]), bf16_to_float(C_ref[m * ldc + n]), 1e-3);
}
TEST_F(bgemmBF16BF16BF16FusedTest, ComputeBiasAddReLU) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] = float_to_bf16(relu(bf16_to_float(C_ref[m * ldc + n]) + bf16_to_float(bias[n])));
    xdnn_bgemm_bf16bf16bf16_compute_biasadd_relu(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc, bias.data());
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(bf16_to_float(C[m * ldc + n]), bf16_to_float(C_ref[m * ldc + n]), 1e-3);
}
TEST_F(bgemmBF16BF16BF16FusedTest, ComputeResidential) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] = float_to_bf16(bf16_to_float(C_ref[m * ldc + n]) + bf16_to_float(bias[n]) + bf16_to_float(res[m * ldc + n]));
    xdnn_bgemm_bf16bf16bf16_compute_residential(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc, bias.data(), res.data(), ldc);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(bf16_to_float(C[m * ldc + n]), bf16_to_float(C_ref[m * ldc + n]), 1e-3);
}
TEST_F(bgemmBF16BF16BF16FusedTest, ComputeResExt) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] = float_to_bf16(bf16_to_float(C_ref[m * ldc + n]) + bf16_to_float(bias[n]) + gamma * bf16_to_float(res[m * ldc + n]));
    xdnn_bgemm_bf16bf16bf16_compute_resext(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc, bias.data(), gamma, res.data(), ldc);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(bf16_to_float(C[m * ldc + n]), bf16_to_float(C_ref[m * ldc + n]), 1e-3);
}
TEST_F(bgemmBF16BF16BF16FusedTest, ComputeResMul) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] = float_to_bf16(bf16_to_float(C_ref[m * ldc + n]) * bf16_to_float(res[m * ldc + n]));
    xdnn_bgemm_bf16bf16bf16_compute_resmul(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc, res.data(), ldc);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(bf16_to_float(C[m * ldc + n]), bf16_to_float(C_ref[m * ldc + n]), 1e-3);
}
// Small bgemm test
TEST(bgemm_BF16BF16BF16, Smallbgemm) {
    int M = 1, N = 1, K = 1;
    XDNN_BF16 a = float_to_bf16(2.0f);
    XDNN_BF16 b = float_to_bf16(3.0f);
    XDNN_BF16 c = float_to_bf16(0.0f);
    XDNN_BF16 c_ref = float_to_bf16(0.0f);
    c_ref = float_to_bf16(bf16_to_float(a) * bf16_to_float(b));
    small_bgemm_bf16bf16bf16(M, N, K, &a, K, &b, N, &c, N);
    EXPECT_NEAR(bf16_to_float(c), bf16_to_float(c_ref), 1e-3);
}
