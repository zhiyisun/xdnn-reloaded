#include "bgemm_f32bf16f32.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

// Helper to fill matrix with a pattern
void fill_fp16_matrix(std::vector<XDNN_BF16>& mat, int rows, int cols, int ld, float start = 0.0f) {
    mat.resize(ld * rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            mat[r * ld + c] = static_cast<XDNN_BF16>(start + r * cols + c);
}

// Reference: packb (row-major KxN output)
void reference_packb(bool transB, int N, int K, const XDNN_BF16* B, int ldb, XDNN_BF16* packedB, int block_rows, int block_cols) {
    std::vector<XDNN_BF16> B_buf;
    const XDNN_BF16* B_used = B;
    if (transB) {
        // Transpose B (original shape KxN, ldb)
        B_buf.resize(K * N, 0);
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < K; ++c) {
                B_buf[c * N + r] = B[r * K + c];
            }
        }
        B_used = B_buf.data();
    }
    int idx = 0;
    int packed_idx = 0;
    int packed_row_per_block = 0;
    if ((K / 2) > block_rows) {
        packed_row_per_block = K / 2;
    }
    else {
        packed_row_per_block = block_rows;
    }

    int packed_cols = block_cols * 2;
    int packed_rows_per_rowB = N / block_cols;

    for (int row = 0; row < K; ++ row) {
        for (int col = 0; col < N; ++ col) {
            idx = row * N + col;
            int pos_in_packed_row = 2 * (idx % block_cols) + (idx / N) % 2;
            int block_per_rowB = (idx % N) / block_cols;
            int packed_row_block_offset = block_per_rowB * packed_row_per_block;
            int packed_row_offset_in_block = idx / (N * 2);
            packed_idx = pos_in_packed_row + (packed_row_block_offset + packed_row_offset_in_block) * block_cols * 2;
            packedB[packed_idx] = B_used[idx];
        }
    }
}

// Parameterized test for packb
struct PackBParams { int N, K, ldb; bool transB; };
class bgemmF32BF16F32PackBTest : public ::testing::TestWithParam<PackBParams> {};
TEST_P(bgemmF32BF16F32PackBTest, PackBCorrectness) {
    auto p = GetParam();
    int block_rows = 16;
    int block_cols = 64;
    int rowsB = p.transB ? p.N : p.K;
    int colsB = p.transB ? p.K : p.N;
    int packed_row = 0;
    if ((p.K / 2) > block_rows) {
        packed_row = (p.K / 2) * p.N / block_cols;
    } else {
        packed_row = block_rows * p.N / block_cols;
    }
    std::vector<XDNN_BF16> B(p.ldb * (p.transB ? p.N : p.K));
    std::vector<XDNN_BF16> packedB(p.ldb * (p.transB ? p.N : p.K), 0);
    std::vector<XDNN_BF16> packedB_ref(p.ldb * (p.transB ? p.N : p.K), 0);
    fill_fp16_matrix(B, p.transB ? p.N : p.K, p.transB ? p.K : p.N, p.ldb);
    reference_packb(p.transB, p.N, p.K, B.data(), p.ldb, packedB_ref.data(), block_rows, block_cols);
    xdnn_bgemm_f32bf16f32_packb(p.transB, p.N, p.K, B.data(), p.ldb, packedB.data(), block_rows, block_cols);

    for (int i = 0; i < p.N * p.K; ++i) EXPECT_EQ(packedB[i], packedB_ref[i]);
}

INSTANTIATE_TEST_SUITE_P(PackB, bgemmF32BF16F32PackBTest, ::testing::Values(
    PackBParams{64, 64, 64, false}, 
    PackBParams{64, 64, 64, true}, 
    PackBParams{128, 64, 128, false},
    PackBParams{192, 48, 192, false},
    PackBParams{192, 128,  128, true},
    PackBParams{192, 128,  128, true},
    PackBParams{192, 128,  128, true},
    PackBParams{4096, 1024, 4096, false},
    PackBParams{1024, 2048, 1024, false},
    PackBParams{6144, 1024, 6144, false},
    PackBParams{1024, 3072, 1024, false},
    PackBParams{151936, 1024, 1024, true}
));
