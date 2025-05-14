#pragma once

#include "data_types/data_types.h"

extern "C" {

int xdnn_small_amx_sgemm_f16f16f16_packb_size(int N, int K, int block_rows, int block_cols);

void xdnn_small_amx_sgemm_f16f16f16_packb(bool transB, int N, int K, const XDNN_FP16 *B, int stride, XDNN_FP16 *packedB,
                                          int size);

// ldb: leading dimension of B (K value when calling pack function)
void xdnn_small_amx_sgemm_f16f16f16_compute(int M, int N, int K, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                            int ldb, XDNN_FP16 *C, int ldc, float beta);
}