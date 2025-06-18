#include "conversion.h"
#include "amx_sgemm_f16f16f16.h"
#include "intrinsic_ext.h"
#include "debug_print.h"
#include <cstring>
#include <immintrin.h>
#include <algorithm>

// AMX SGEMM operations for FP16 data types

int xdnn_small_amx_sgemm_f16f16f16_packb_size(int N, int K, int block_rows, int block_cols) {
    DEBUG_PRINT();
    return 0;
}

void xdnn_small_amx_sgemm_f16f16f16_packb(bool transB, int N, int K, const XDNN_FP16 *B, int stride, XDNN_FP16 *packedB,
                                          int size) {
    DEBUG_PRINT();
}

// AMX optimized GEMM computation for FP16 input and output
void xdnn_small_amx_sgemm_f16f16f16_compute(int M, int N, int K, const XDNN_FP16 *A, int lda, const XDNN_FP16 *packedB,
                                            int ldb, XDNN_FP16 *C, int ldc, float beta) {
    DEBUG_PRINT();
}
