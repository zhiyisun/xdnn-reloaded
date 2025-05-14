#pragma once

#include "data_types/data_types.h"

extern "C" {

int xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(int N, int K, int block_rows, int block_cols);

void xdnn_small_amx_sgemm_bf16bf16bf16_packb(
        bool transB, int N, int K, const XDNN_BF16 *B, int stride, XDNN_BF16 *packedB, int size);

// // ldb: leading dimension of B (K value when calling pack function)
void xdnn_small_amx_sgemm_bf16bf16bf16_compute(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_BF16 *packedB, int ldb, XDNN_BF16 *C, int ldc, float beta);

void xdnn_small_amx_sgemm_bf16bf16f32_compute(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_BF16 *packedB, int ldb, float *C, int ldc, float beta);

void xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a(int M, int N, int K, const XDNN_BF16 *A,
        int lda, const XDNN_BF16 *packedB, XDNN_BF16 *C, int ldc, float alpha, float beta);

void xdnn_small_amx_sgemm_bf16bf16f32_compute_BA16a64b2a(int M, int N, int K, const XDNN_BF16 *A,
        int lda, const XDNN_BF16 *packedB, float *C, int ldc, float alpha, float beta);

// To perform Batched BF16 matrix multiplication
// Batch C = A * Bacth B
// M: number of rows in matrix A, now only support M=1
// N: number of columns in matrix B, now N must be a multiple of 64
// K: number of columns in matrix A and rows in matrix B, now K must be a multiple of 64
// A: input matrix A, data type is BF16
// lda: leading dimension of matrix A
// packedBBatch: Batch of Packed matrix B, data type is FP8_E4M3
// C: output matrix C, data type is BF16
// ldc: leading dimension of matrix C
// alphaBatch: Scaling factors for each matrix B, data type is float
// packedBBatchSize: the Batch size of the packed matrix B
void xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a_batch_C(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_BF16 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *alphaBatch, int packedBBatchSize);

// To perform Batched FP8 matrix multiplication
// Batch C = A * Bacth B
// M: number of rows in matrix A, now only support M=1
// NBatch: number of columns in matrix Bs, now N must be a multiple of 64
// K: number of columns in matrix A and rows in matrix B, now K must be a multiple of 64
// A: input matrix A, data type is BF16
// lda: leading dimension of matrix A
// packedBBatch: Batch of Packed matrix B, data type is FP8_E4M3
// C: output matrix C, data type is BF16
// ldc: leading dimension of matrix C
// alphaBatch: Scaling factors for each matrix B, data type is float
// packedBBatchSize: the Batch size of the packed matrix B
void xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a_batch_CM(int M, const int *NBatch, int K, const XDNN_BF16 *A,
        int lda, const XDNN_BF16 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *alphaBatch,
        int packedBBatchSize);

// To perform Batched FP8 matrix multiplication
// C = Sum(Batch A * Bacth B)
// M: number of rows in matrix A, now only support M=1
// N: number of columns in matrix B, now N must be a multiple of 64
// K: number of columns in matrix A and rows in matrix B, now K must be a multiple of 64
// ABatch: Batch of input matrix A, data type is BF16
// ldab: Batch of leading dimension of matrix As
// packedBBatch: Batch of Packed matrix B, data type is FP8_E4M3
// C: output matrix C, data type is BF16
// ldc: leading dimension of matrix C
// alphaBatch: Scaling factors for each matrix B, data type is float
// packedBBatchSize: the Batch size of the packed matrix B
void xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a_batch_A(int M, int N, int K, const XDNN_BF16 *ABatch[],
        const int *ldab, const XDNN_BF16 *packedBBatch[], XDNN_BF16 *C, int ldc, const float *alphaBatch,
        int packedBBatchSize);

// To perform Batched FP8 matrix multiplication
// C = Sum(Batch A * Bacth B)
// M: number of rows in matrix A, now only support M=1
// N: number of columns in matrix B, now N must be a multiple of 64
// KBatch: number of columns in matrix As and rows in matrix Bs, now K must be a multiple of 64
// ABatch: Batch of input matrix A, data type is BF16
// ldab: Batch of leading dimension of matrix As
// packedBBatch: Batch of Packed matrix B, data type is FP8_E4M3
// C: output matrix C, data type is BF16
// ldc: leading dimension of matrix C
// scaleBBatch: Scaling factors for matrix B, data type is float
// ldsbb: Stride for scaling factors
// blockSize: size of the block to mapping matrix B and scale factors
// alphaBatch: Scaling factors for each matrix B, data type is float
// packedBBatchSize: the Batch size of the packed matrix B

void xdnn_small_amx_sgemm_bf16bf16bf16_compute_BA16a64b2a_batch_AM(int M, int N, const int *KBatch,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_BF16 *packedBBatch[], XDNN_BF16 *C, int ldc,
        const float *alphaBatch, int packedBBatchSize);

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_test_all(int option, int M, const int *NBatch, const int *KBatch,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_BF16 *packedBBatch[], XDNN_BF16 *C16[], float *C32[],
        const int *ldcb, const float *alphaBatch, int packedBBatchSize, int layers);
}