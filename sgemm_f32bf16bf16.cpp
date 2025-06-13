#include "sgemm_f32bf16bf16.h"
#include "debug_print.h"
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <stdexcept>

// Function to perform matrix multiplication C = A * B
// where A is F32, B is BF16, and C is BF16
// Only supports transB=false, M=1 for the next token calculation
// in transformer attention mechanism
void small_sgemm_f32bf16bf16(bool transB, int M, int N, int K, const float *A, int lda, const XDNN_BF16 *B, int ldb, XDNN_BF16 *C, int ldc) {
    DEBUG_PRINT();
}

/**
 * This function is specially designed for paged attention
 * Matrix B is like (blockSize=4):
 *                                   |<---- ldb ----->|
 *  ________________ ________________|_h0_|___________|_h0_|___________
 * | #head*headSize | #head*headSize |    |           |                | block0
 * |________________|________________|____|___________|________________|
 * |                |                |                |                | block1
 * |________________|________________|________________|________________|
 * |                |                |                |                | block2
 * |________________|________________|________________|________________|
 * |<--------------------------- blockStride ------------------------->|
 */
void small_sgemm_f32bf16bf16_b(bool transB, int M, int N, int K, const float *A, int lda, 
                               const XDNN_BF16 *B, int ldb, XDNN_BF16 *C, int ldc, 
                               int *blockIndices, int blockStride, int blockSize) {
    DEBUG_PRINT();
}
