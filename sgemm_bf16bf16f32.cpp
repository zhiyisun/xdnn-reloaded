#include "sgemm_bf16bf16f32.h"
#include "intrinsic_ext.h"
#include "debug_print.h"
#include <immintrin.h>
#include <cstring>

/**
 * Implementation for standard small_sgemm_bf16bf16f32 function.
 * This function performs matrix multiplication: C = A * B
 * where:
 * - A is an M x K matrix in BF16 format
 * - B is a K x N matrix in BF16 format (if transB=false) or N x K matrix (if transB=true)
 * - C is an M x N matrix in F32 format
 * 
 * As specified in the header, this function assumes:
 * - M = 1 (single output row)
 * - transB = true (B is in column-major order)
 */
void small_sgemm_bf16bf16f32(bool transB, int M, int N, int K, 
                            const XDNN_BF16 *A, int lda, 
                            const XDNN_BF16 *B, int ldb, 
                            float *C, int ldc) {
    DEBUG_PRINT();
}

/**
 * Implementation for paged attention version of small_sgemm_bf16bf16f32_b.
 * This function performs matrix multiplication where B is organized in blocks:
 * C = A * B
 * where:
 * - A is an M x K matrix in BF16 format
 * - B is organized in blocks with strides defined by blockStride and blockSize
 * - C is an M x N matrix in F32 format
 * 
 * As specified in the header, this function assumes:
 * - M = 1 (single output row)
 * - transB = true (each block of B is in column-major order)
 */
void small_sgemm_bf16bf16f32_b(bool transB, int M, int N, int K, 
                              const XDNN_BF16 *A, int lda, 
                              const XDNN_BF16 *B, int ldb, 
                              float *C, int ldc, 
                              int *blockIndices, int blockStride, int blockSize) {
    DEBUG_PRINT();
}
