#include "sgemm_f32f16bf16.h"
#include "debug_print.h"
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <stdexcept>

// Function to perform matrix multiplication C = alpha * A * B + beta * C
// where A is F32, B is FP16, and C is BF16
// Only supports transB=false, M=1, alpha=1, beta=0|1 for the next token calculation
// in transformer attention mechanism
void small_sgemm_f32f16bf16(bool transB, int M, int N, int K, float alpha, const float *A, int lda, const XDNN_FP16 *B, int ldb, float beta, XDNN_BF16 *C, int ldc) {
    DEBUG_PRINT();
}