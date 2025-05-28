#include "hgemm_f32f16f32.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

// Reference: simple float16 emulation for test (just use float for now)
using RefFP16 = float;

// Helper to fill matrix with a pattern
void fill_fp16_matrix(std::vector<XDNN_FP16>& mat, int rows, int cols, int ld, float start = 1.0f) {
    mat.resize(ld * rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            mat[r * ld + c] = static_cast<XDNN_FP16>(start + r * cols + c);
}

// Reference: packb (row-major KxN output)
void reference_packb(bool transB, int N, int K, const XDNN_FP16* B, int ldb, XDNN_FP16* packedB) {
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            int src_idx = transB ? (n * ldb + k) : (k * ldb + n);
            int dst_idx = k * N + n;
            packedB[dst_idx] = B[src_idx];
        }
    }
}

// Reference: C = alpha * A * B + beta * C (B is KxN row-major, A is MxK row-major)
void reference_compute(bool transA, int M, int N, int K, float alpha, const XDNN_FP16* A, int lda, const XDNN_FP16* packedB, float beta, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = static_cast<float>(transA ? A[k * lda + m] : A[m * lda + k]);
                float b_val = static_cast<float>(packedB[k * N + n]);
                sum += a_val * b_val;
            }
            if (beta == 0.0f) C[m * ldc + n] = alpha * sum;
            else C[m * ldc + n] = alpha * sum + C[m * ldc + n];
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
class HgemmF32F16F32PackBTest : public ::testing::TestWithParam<PackBParams> {};
TEST_P(HgemmF32F16F32PackBTest, PackBCorrectness) {
    auto p = GetParam();
    std::vector<XDNN_FP16> B(p.ldb * (p.transB ? p.N : p.K));
    std::vector<XDNN_FP16> packedB(p.K * p.N, 0), packedB_ref(p.K * p.N, 0);
    fill_fp16_matrix(B, p.transB ? p.N : p.K, p.transB ? p.K : p.N, p.ldb);
    reference_packb(p.transB, p.N, p.K, B.data(), p.ldb, packedB_ref.data());
    xdnn_hgemm_f32f16f32_packb(p.transB, p.N, p.K, B.data(), p.ldb, packedB.data());
    for (int i = 0; i < p.K * p.N; ++i) EXPECT_EQ(packedB[i], packedB_ref[i]);
}
INSTANTIATE_TEST_SUITE_P(PackB, HgemmF32F16F32PackBTest, ::testing::Values(
    PackBParams{4, 4, 4, false}, 
    PackBParams{4, 4, 4, true}, 
    PackBParams{8, 8, 10, false}
));

// Parameterized test for compute
struct ComputeParams { int M, N, K; float alpha, beta; bool transA; };
class HgemmF16F16F32ComputeTest : public ::testing::TestWithParam<ComputeParams> {};
TEST_P(HgemmF16F16F32ComputeTest, ComputeCorrectness) {
    auto p = GetParam();
    std::vector<XDNN_FP16> A(p.M * p.K), packedB(p.K * p.N);
    std::vector<float> C(p.M * p.N, 1.23f), C_ref(p.M * p.N, 1.23f);
    fill_fp16_matrix(A, p.M, p.K, p.K);
    fill_fp16_matrix(packedB, p.K, p.N, p.N);
    reference_compute(p.transA, p.M, p.N, p.K, p.alpha, A.data(), p.K, packedB.data(), p.beta, C_ref.data(), p.N);
    xdnn_hgemm_f16f16f32_compute(p.transA, p.M, p.N, p.K, p.alpha, A.data(), p.K, packedB.data(), p.beta, C.data(), p.N);
    for (int i = 0; i < p.M * p.N; ++i) EXPECT_NEAR(C[i], C_ref[i], 1e-3);
}
INSTANTIATE_TEST_SUITE_P(Compute, HgemmF16F16F32ComputeTest, ::testing::Values(
    ComputeParams{2, 2, 2, 1.0f, 0.0f, false}, 
    ComputeParams{2, 2, 2, 1.0f, 1.0f, false},
    ComputeParams{4, 4, 4, 1.0f, 0.0f, false}, 
    ComputeParams{4, 4, 4, 1.0f, 0.0f, false}
));

// Fused op tests (SiLU, GELU, BiasAdd, BiasAddReLU, Residential, ResExt, ResMul)
class HgemmF16F16F32FusedTest : public ::testing::Test {
protected:
    int M = 16, N = 16, K = 16;
    int ldc = 16; // Use N for now, but can be larger for padding tests
    std::vector<XDNN_FP16> A, packedB;
    std::vector<float> C, C_ref, bias, res;
    float gamma = 1.5f;
    void SetUp() override {
        A.resize(M * K); fill_fp16_matrix(A, M, K, K);
        packedB.resize(K * N); fill_fp16_matrix(packedB, K, N, N);
        C.assign(M * ldc, 1.23f); C_ref = C;
        bias.assign(N, 2.0f);
        res.assign(M * ldc, 3.0f);
    }
};
TEST_F(HgemmF16F16F32FusedTest, ComputeSiLU) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] = silu(C_ref[m * ldc + n]);
    xdnn_hgemm_f16f16f32_compute_silu(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(C[m * ldc + n], C_ref[m * ldc + n], 1e-3);
}
TEST_F(HgemmF16F16F32FusedTest, ComputeGELU) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] = gelu(C_ref[m * ldc + n]);
    xdnn_hgemm_f16f16f32_compute_gelu(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(C[m * ldc + n], C_ref[m * ldc + n], 1e-3);
}
TEST_F(HgemmF16F16F32FusedTest, ComputeBiasAdd) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] += bias[n];
    xdnn_hgemm_f16f16f32_compute_biasadd(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc, bias.data());
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(C[m * ldc + n], C_ref[m * ldc + n], 1e-3);
}
TEST_F(HgemmF16F16F32FusedTest, ComputeBiasAddReLU) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] = relu(C_ref[m * ldc + n] + bias[n]);
    xdnn_hgemm_f16f16f32_compute_biasadd_relu(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc, bias.data());
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(C[m * ldc + n], C_ref[m * ldc + n], 1e-3);
}
TEST_F(HgemmF16F16F32FusedTest, ComputeResidential) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] += bias[n] + res[m * ldc + n];
    xdnn_hgemm_f16f16f32_compute_residential(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc, bias.data(), res.data(), ldc);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(C[m * ldc + n], C_ref[m * ldc + n], 1e-3);
}
TEST_F(HgemmF16F16F32FusedTest, ComputeResExt) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] += bias[n] + gamma * res[m * ldc + n];
    xdnn_hgemm_f16f16f32_compute_resext(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc, bias.data(), gamma, res.data(), ldc);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(C[m * ldc + n], C_ref[m * ldc + n], 1e-3);
}
TEST_F(HgemmF16F16F32FusedTest, ComputeResMul) {
    reference_compute(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C_ref.data(), ldc);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) C_ref[m * ldc + n] *= res[m * ldc + n];
    xdnn_hgemm_f16f16f32_compute_resmul(false, M, N, K, 1.0f, A.data(), K, packedB.data(), 0.0f, C.data(), ldc, res.data(), ldc);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            EXPECT_NEAR(C[m * ldc + n], C_ref[m * ldc + n], 1e-3);
}
// Small SGEMM test
TEST(SGEMM_F32F16F32, SmallSGEMM) {
    int M = 1, N = 1, K = 1;
    float a = 2.0f;
    XDNN_FP16 b = 3.0f;
    float c = 0.0f;
    float c_ref = 0.0f;
    c_ref = a * static_cast<float>(b);
    small_hgemm_f32f16f32(M, N, K, &a, K, &b, N, &c, N);
    EXPECT_NEAR(c, c_ref, 1e-3);
}
