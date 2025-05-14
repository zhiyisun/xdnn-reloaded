#pragma once

#include "data_types/data_types.h"

extern "C" {
// To calculate the size needed to pack matrix B
// pack_size is 64 for AMX
// First to call this and prepare packedB memory
// Then call xdnn_small_amx_sgemm_bf16f8bf16_packb
int xdnn_small_amx_sgemm_bf16f8bf16_packb_size(int N, int K, int pack_size);

// To pack matrix B
// transB: if the input 'b' is transposed or not
// B is in K x N if transB = false
// B is in N x K if transB = true
// pack_size is 64 for AMX
void xdnn_small_amx_sgemm_bf16f8bf16_packb(
        bool transB, int N, int K, const XDNN_E4M3 *B, int ldb, XDNN_E4M3 *packedB, int pack_size);

// To perform FP8 matrix multiplication
// single-thread version xdnn_small_amx_sgemm_bf16f8bf16_compute_single
// multi-thread version xdnn_small_amx_sgemm_bf16f8bf16_compute
// M: number of rows in matrix A, now only support M=1
// N: number of columns in matrix B, now N must be a multiple of 64
// K: number of columns in matrix A and rows in matrix B, now K must be a multiple of 64
// A: input matrix A, data type is BF16
// lda: leading dimension of matrix A
// packedB: packed matrix B, data type is FP8_E4M3
// C: output matrix C, data type is BF16
// ldc: leading dimension of matrix C
// scaleB: scaling factors for matrix B, data type is float
// lds: stride for scaling factors
// blockSize: size of the block to mapping matrix B and scale factors
// alpha: scaling factor for matrix multiplication
// beta: scaling factor for matrix addition
// bias: bias values
void xdnn_small_amx_sgemm_bf16f8bf16_compute_single(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_E4M3 *packedB, XDNN_BF16 *C, int ldc, const float *scaleB, int lds, int blockSize, float alpha,
        float beta, const float *bias);
void xdnn_small_amx_sgemm_bf16f8bf16_compute(int M, int N, int K, const XDNN_BF16 *A, int lda, const XDNN_E4M3 *packedB,
        XDNN_BF16 *C, int ldc, const float *scaleB, int lds, int blockSize, float alpha, float beta, const float *bias);

// To perform Batched FP8 matrix multiplication
// Batch C = A * Bacth B
// M: number of rows in matrix A, now only support M=1
// N: number of columns in matrix B, now N must be a multiple of 64
// K: number of columns in matrix A and rows in matrix B, now K must be a multiple of 64
// A: input matrix A, data type is BF16
// lda: leading dimension of matrix A
// packedBBatch: Batch of Packed matrix B, data type is FP8_E4M3
// C: output matrix C, data type is BF16
// ldc: leading dimension of matrix C
// scaleBBatch: Scaling factors for matrix B, data type is float
// ldsbb: Stride for scaling factors
// blockSize: size of the block to mapping matrix B and scale factors
// alphaBatch: Scaling factors for each matrix B, data type is float
// packedBBatchSize: the Batch size of the packed matrix B
void xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_C(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize);

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
// scaleBBatch: Scaling factors for matrix B, data type is float
// ldsbb: Stride for scaling factors
// blockSize: size of the block to mapping matrix B and scale factors
// alphaBatch: Scaling factors for each matrix B, data type is float
// packedBBatchSize: the Batch size of the packed matrix B
void xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_CM(int M, const int *NBatch, int K, const XDNN_BF16 *A, int lda,
        const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize);

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
// scaleBBatch: Scaling factors for matrix B, data type is float
// ldsbb: Stride for scaling factors
// blockSize: size of the block to mapping matrix B and scale factors
// alphaBatch: Scaling factors for each matrix B, data type is float
// packedBBatchSize: the Batch size of the packed matrix B
void xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_A(int M, int N, int K, const XDNN_BF16 *ABatch[], const int *ldab,
        const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc, const float *scaleBBatch[], const int *scaleB_lda,
        int blockSize, const float *alphaBatch, int packedBBatchSize);

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
void xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_AM(int M, int N, const int *KBatch, const XDNN_BF16 *ABatch[],
        const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize);

// To perform FP8 matrix multiplication with residential
// M: number of rows in matrix A, now only support M=1
// N: number of columns in matrix B, now N must be a multiple of 64
// K: number of columns in matrix A and rows in matrix B, now K must be a multiple of 64
// A: input matrix A, data type is BF16
// lda: leading dimension of matrix A
// packedB: packed matrix B, data type is FP8_E4M3
// C: output matrix C, data type is BF16
// ldc: leading dimension of matrix C
// scaleB: scaling factors for matrix B, data type is float
// lds: stride for scaling factors
// blockSize: size of the block to mapping matrix B and scale factors
// alpha: scaling factor for matrix multiplication
// beta: scaling factor for matrix addition
// bias: bias values
// res: residential values
// ldres: stride for residential values
void xdnn_small_amx_sgemm_bf16f8bf16_compute_residential(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_E4M3 *packedB, XDNN_BF16 *C, int ldc, const float *scaleB, int lds, int blockSize, float alpha,
        float beta, const float *bias, const XDNN_BF16 *res, int ldres);

// To perform Batched FP8 matrix multiplication with residential
// Batch C = A * Batch B + Batch residential
// M: number of rows in matrix A, now only support M=1
// N: number of columns in matrix B, now N must be a multiple of 64
// K: number of columns in matrix A and rows in matrix B, now K must be a multiple of 64
// A: input matrix A, data type is BF16
// lda: leading dimension of matrix A
// packedBBatch: Batch of Packed matrix B, data type is FP8_E4M3
// CBatch: Batch of output matrix C, data type is BF16
// ldcb: Batch of leading dimension of matrix Cs
// scaleBBatch: Scaling factors for matrix B, data type is float
// ldsbb: Stride for scaling factors
// blockSize: size of the block to mapping matrix B and scale factors
// alphaBatch: Scaling factors for each matrix B, data type is float
// packedBBatchSize: the Batch size of the packed matrix B
// resBatch: Batch of residential values
// ldresb: Batch of stride for residential values
void xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_C(int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *resBatch[], const int *ldresb);

// To perform Batched FP8 matrix multiplication with residential
// Batch C = A * Batch B + Batch residential
// M: number of rows in matrix A, now only support M=1
// NBatch: number of columns in matrix B, now N must be a multiple of 64
// K: number of columns in matrix A and rows in matrix B, now K must be a multiple of 64
// A: input matrix A, data type is BF16
// lda: leading dimension of matrix A
// packedBBatch: Batch of Packed matrix B, data type is FP8_E4M3
// CBatch: Batch of output matrix C, data type is BF16
// ldcb: Batch of leading dimension of matrix Cs
// scaleBBatch: Scaling factors for matrix B, data type is float
// ldsbb: Stride for scaling factors
// blockSize: size of the block to mapping matrix B and scale factors
// alphaBatch: Scaling factors for each matrix B, data type is float
// packedBBatchSize: the Batch size of the packed matrix B
// resBatch: Batch of residential values
// ldresb: Batch of stride for residential values
void xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_CM(int M, const int *NBatch, int K, const XDNN_BF16 *A,
        int lda, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *resBatch[], const int *ldresb);

// To perform Batched FP8 matrix multiplication with residential
// C = Sum(Batch A * Bacth B) + residential
// M: number of rows in matrix A, now only support M=1
// N: number of columns in matrix B, now N must be a multiple of 64
// K: number of columns in matrix A and rows in matrix B, now K must be a multiple of 64
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
// res: residential values
// ldres: stride for residential values
void xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_A(int M, int N, int K, const XDNN_BF16 *ABatch[],
        const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize, const XDNN_BF16 *res,
        int ldres);

void xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_AM(int M, int N, const int *KBatch,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *res, int ldres);

// Function to perform matrix multiplication test
// caseid: identifier for the backend implementation.
const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_test(int caseid, int M, int N, int K, const XDNN_BF16 *A, int lda,
        const XDNN_E4M3 *packedB, XDNN_BF16 *C, int ldc, const float *scaleB, int lds, int blockSize, float alpha,
        float beta, const float *bias);

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_test(int caseid, int M, int N, int K,
        const XDNN_BF16 *A, int lda, const XDNN_E4M3 *packedB, XDNN_BF16 *C, int ldc, const float *scaleB, int lds,
        int blockSize, float alpha, float beta, const float *bias, const XDNN_BF16 *res, int ldres);

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_A_test(int caseid, int M, int N, int K,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int BSize);

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_AM_test(int caseid, int M, int N, const int *KBatch,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int BSize);

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_A_test(int caseid, int M, int N, int K,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *res, int ldres);

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_AM_test(int caseid, int M, int N, int *KBatch,
        const XDNN_BF16 *ABatch[], const int *ldab, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *C, int ldc,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *res, int ldres);

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_C_test(int caseid, int M, int N, int K, const XDNN_BF16 *A,
        int lda, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb, const float *scaleBBatch[],
        const int *scaleB_lda, int blockSize, const float *alphaBatch, int BSize);

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_batch_CM_test(int caseid, int M, const int *NBatch, int K,
        const XDNN_BF16 *A, int lda, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int BSize);

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_C_test(int caseid, int M, int N, int K,
        const XDNN_BF16 *A, int lda, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *resBatch[], const int *ldresb);

const char *xdnn_small_amx_sgemm_bf16f8bf16_compute_residential_batch_CM_test(int caseid, int M, int *NBatch, int K,
        const XDNN_BF16 *A, int lda, const XDNN_E4M3 *packedBBatch[], XDNN_BF16 *CBatch[], const int *ldcb,
        const float *scaleBBatch[], const int *scaleB_lda, int blockSize, const float *alphaBatch, int packedBBatchSize,
        const XDNN_BF16 *resBatch[], const int *ldresb);
}