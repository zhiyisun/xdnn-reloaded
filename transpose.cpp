#include "conversion.h"
#include "transpose.h"
#include "platform_detection.h"
#include <cstring>
#include <cassert>

namespace {
// Generic transpose implementation for any data type
template<typename T>
void transpose_generic(const T *src, int src_rows, int src_cols, int src_stride, 
                       T *dst, int dst_stride) {
    for (int i = 0; i < src_rows; i++) {
        for (int j = 0; j < src_cols; j++) {
            dst[j * dst_stride + i] = src[i * src_stride + j];
        }
    }
}

// SSE-optimized transpose for float32 (4x4 blocks)
void transpose_sse_float_4x4(const float *src, int src_stride, float *dst, int dst_stride) {
    // Load 4x4 block from source
    __m128 row0 = _mm_loadu_ps(&src[0 * src_stride]);
    __m128 row1 = _mm_loadu_ps(&src[1 * src_stride]);
    __m128 row2 = _mm_loadu_ps(&src[2 * src_stride]);
    __m128 row3 = _mm_loadu_ps(&src[3 * src_stride]);

    // Transpose 4x4 matrix
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

    // Store transposed data
    _mm_storeu_ps(&dst[0 * dst_stride], row0);
    _mm_storeu_ps(&dst[1 * dst_stride], row1);
    _mm_storeu_ps(&dst[2 * dst_stride], row2);
    _mm_storeu_ps(&dst[3 * dst_stride], row3);
}

// SSE-optimized transpose for float32 matrices
void transpose_sse_float(const float *src, int src_rows, int src_cols, int src_stride, 
                         float *dst, int dst_stride) {
    // Process blocks of 4x4
    int block_rows = (src_rows / 4) * 4;
    int block_cols = (src_cols / 4) * 4;

    // Process full 4x4 blocks
    for (int i = 0; i < block_rows; i += 4) {
        for (int j = 0; j < block_cols; j += 4) {
            transpose_sse_float_4x4(&src[i * src_stride + j], src_stride,
                                   &dst[j * dst_stride + i], dst_stride);
        }
    }

    // Handle remaining rows
    for (int i = block_rows; i < src_rows; i++) {
        for (int j = 0; j < src_cols; j++) {
            dst[j * dst_stride + i] = src[i * src_stride + j];
        }
    }

    // Handle remaining columns
    for (int i = 0; i < block_rows; i++) {
        for (int j = block_cols; j < src_cols; j++) {
            dst[j * dst_stride + i] = src[i * src_stride + j];
        }
    }
}

// AVX-optimized transpose for float32 (8x8 blocks)
void transpose_avx_float_8x8(const float *src, int src_stride, float *dst, int dst_stride) {
#ifdef __AVX__
    // Load 8x8 block from source
    __m256 row0 = _mm256_loadu_ps(&src[0 * src_stride]);
    __m256 row1 = _mm256_loadu_ps(&src[1 * src_stride]);
    __m256 row2 = _mm256_loadu_ps(&src[2 * src_stride]);
    __m256 row3 = _mm256_loadu_ps(&src[3 * src_stride]);
    __m256 row4 = _mm256_loadu_ps(&src[4 * src_stride]);
    __m256 row5 = _mm256_loadu_ps(&src[5 * src_stride]);
    __m256 row6 = _mm256_loadu_ps(&src[6 * src_stride]);
    __m256 row7 = _mm256_loadu_ps(&src[7 * src_stride]);

    // Transpose 8x8 matrix (AVX doesn't have a direct transpose instruction like SSE)
    // We need to do it manually using permutes and shuffles
    
    // First, interleave 2x2 blocks
    __m256 ab_0145 = _mm256_unpacklo_ps(row0, row1); // a0,b0,a1,b1,a4,b4,a5,b5
    __m256 ab_2367 = _mm256_unpacklo_ps(row2, row3); // a2,b2,a3,b3,a6,b6,a7,b7
    __m256 cd_0145 = _mm256_unpackhi_ps(row0, row1); // c0,d0,c1,d1,c4,d4,c5,d5
    __m256 cd_2367 = _mm256_unpackhi_ps(row2, row3); // c2,d2,c3,d3,c6,d6,c7,d7
    
    __m256 ef_0145 = _mm256_unpacklo_ps(row4, row5); // e0,f0,e1,f1,e4,f4,e5,f5
    __m256 ef_2367 = _mm256_unpacklo_ps(row6, row7); // e2,f2,e3,f3,e6,f6,e7,f7
    __m256 gh_0145 = _mm256_unpackhi_ps(row4, row5); // g0,h0,g1,h1,g4,h4,g5,h5
    __m256 gh_2367 = _mm256_unpackhi_ps(row6, row7); // g2,h2,g3,h3,g6,h6,g7,h7
    
    // Now interleave 4x4 blocks
    __m256 abcd_01 = _mm256_shuffle_ps(ab_0145, ab_2367, 0x44); // a0,b0,a2,b2,a4,b4,a6,b6
    __m256 abcd_23 = _mm256_shuffle_ps(ab_0145, ab_2367, 0xEE); // a1,b1,a3,b3,a5,b5,a7,b7
    __m256 abcd_45 = _mm256_shuffle_ps(cd_0145, cd_2367, 0x44); // c0,d0,c2,d2,c4,d4,c6,d6
    __m256 abcd_67 = _mm256_shuffle_ps(cd_0145, cd_2367, 0xEE); // c1,d1,c3,d3,c5,d5,c7,d7
    
    __m256 efgh_01 = _mm256_shuffle_ps(ef_0145, ef_2367, 0x44); // e0,f0,e2,f2,e4,f4,e6,f6
    __m256 efgh_23 = _mm256_shuffle_ps(ef_0145, ef_2367, 0xEE); // e1,f1,e3,f3,e5,f5,e7,f7
    __m256 efgh_45 = _mm256_shuffle_ps(gh_0145, gh_2367, 0x44); // g0,h0,g2,h2,g4,h4,g6,h6
    __m256 efgh_67 = _mm256_shuffle_ps(gh_0145, gh_2367, 0xEE); // g1,h1,g3,h3,g5,h5,g7,h7
    
    // Permute to get final transposed rows
    __m256 transposed0 = _mm256_permute2f128_ps(abcd_01, efgh_01, 0x20);
    __m256 transposed1 = _mm256_permute2f128_ps(abcd_23, efgh_23, 0x20);
    __m256 transposed2 = _mm256_permute2f128_ps(abcd_45, efgh_45, 0x20);
    __m256 transposed3 = _mm256_permute2f128_ps(abcd_67, efgh_67, 0x20);
    __m256 transposed4 = _mm256_permute2f128_ps(abcd_01, efgh_01, 0x31);
    __m256 transposed5 = _mm256_permute2f128_ps(abcd_23, efgh_23, 0x31);
    __m256 transposed6 = _mm256_permute2f128_ps(abcd_45, efgh_45, 0x31);
    __m256 transposed7 = _mm256_permute2f128_ps(abcd_67, efgh_67, 0x31);
    
    // Store the result
    _mm256_storeu_ps(&dst[0 * dst_stride], transposed0);
    _mm256_storeu_ps(&dst[1 * dst_stride], transposed1);
    _mm256_storeu_ps(&dst[2 * dst_stride], transposed2);
    _mm256_storeu_ps(&dst[3 * dst_stride], transposed3);
    _mm256_storeu_ps(&dst[4 * dst_stride], transposed4);
    _mm256_storeu_ps(&dst[5 * dst_stride], transposed5);
    _mm256_storeu_ps(&dst[6 * dst_stride], transposed6);
    _mm256_storeu_ps(&dst[7 * dst_stride], transposed7);
#endif
}

// AVX-optimized transpose for float32 matrices
void transpose_avx_float(const float *src, int src_rows, int src_cols, int src_stride, 
                        float *dst, int dst_stride) {
#ifdef __AVX__
    // Process blocks of 8x8
    int block_rows = (src_rows / 8) * 8;
    int block_cols = (src_cols / 8) * 8;

    // Process full 8x8 blocks
    for (int i = 0; i < block_rows; i += 8) {
        for (int j = 0; j < block_cols; j += 8) {
            transpose_avx_float_8x8(&src[i * src_stride + j], src_stride,
                                   &dst[j * dst_stride + i], dst_stride);
        }
    }

    // Handle remaining rows and columns with SSE or scalar code
    // First handle the case where we have complete 4x4 blocks
    for (int i = block_rows; i < ((src_rows / 4) * 4); i += 4) {
        for (int j = 0; j < block_cols; j += 4) {
            transpose_sse_float_4x4(&src[i * src_stride + j], src_stride,
                                   &dst[j * dst_stride + i], dst_stride);
        }
    }
    
    for (int i = 0; i < block_rows; i += 4) {
        for (int j = block_cols; j < ((src_cols / 4) * 4); j += 4) {
            transpose_sse_float_4x4(&src[i * src_stride + j], src_stride,
                                   &dst[j * dst_stride + i], dst_stride);
        }
    }

    // Handle remaining elements with scalar code
    int aligned_rows = (src_rows / 4) * 4;
    int aligned_cols = (src_cols / 4) * 4;
    
    // Remaining rows
    for (int i = aligned_rows; i < src_rows; i++) {
        for (int j = 0; j < src_cols; j++) {
            dst[j * dst_stride + i] = src[i * src_stride + j];
        }
    }

    // Remaining columns
    for (int i = 0; i < aligned_rows; i++) {
        for (int j = aligned_cols; j < src_cols; j++) {
            dst[j * dst_stride + i] = src[i * src_stride + j];
        }
    }
#endif
}

} // anonymous namespace

// Implementation of various transpose functions
void xdnn_transpose(const float *src, int src_rows, int src_cols, int src_stride, float *dst, int dst_stride) {
    // Use the best implementation based on CPU capabilities
    xdnn::OptimizationLevel level = xdnn::getBestOptimizationLevel();
    
    if (level >= xdnn::OptimizationLevel::AVX) {
        transpose_avx_float(src, src_rows, src_cols, src_stride, dst, dst_stride);
    } else if (level >= xdnn::OptimizationLevel::SSE2) {
        transpose_sse_float(src, src_rows, src_cols, src_stride, dst, dst_stride);
    } else {
        transpose_generic(src, src_rows, src_cols, src_stride, dst, dst_stride);
    }
}

void xdnn_transpose(const XDNN_BF16 *src, int src_rows, int src_cols, int src_stride, XDNN_BF16 *dst, int dst_stride) {
    // For now, use generic implementation for BF16
    // This can be optimized later with AVX-512 when available
    transpose_generic(src, src_rows, src_cols, src_stride, dst, dst_stride);
}

void xdnn_transpose(const int *src, int src_rows, int src_cols, int src_stride, int *dst, int dst_stride) {
    // For integers, use generic implementation for now
    // This can be optimized later
    transpose_generic(src, src_rows, src_cols, src_stride, dst, dst_stride);
}

// Optimized transpose functions for specific sizes
void xdnn_transpose_16x16_v1(const int32_t *src, int src_stride, int32_t *dst, int dst_stride) {
    // 16x16 transpose implementation for int32
    // For now, use generic implementation
    transpose_generic(src, 16, 16, src_stride, dst, dst_stride);
}

void xdnn_transpose_16x16_v2(const int32_t *src, int src_stride, int32_t *dst, int dst_stride) {
    // Alternative 16x16 transpose implementation for int32
    // For now, same as v1
    xdnn_transpose_16x16_v1(src, src_stride, dst, dst_stride);
}

void xdnn_transpose_16xN_v1(const int32_t *src, int cols, int src_stride, int32_t *dst, int dst_stride) {
    // 16xN transpose implementation for int32
    // For now, use generic implementation
    transpose_generic(src, 16, cols, src_stride, dst, dst_stride);
}

// Special packing transposes for BF16
void xdnn_transpose16x32_packBA16a16b2a_v1(const XDNN_BF16 *src, int src_stride, XDNN_BF16 *dst, int dst_stride) {
    // Specialized transpose for BF16 with packing pattern
    // This needs to be implemented with proper optimizations later
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 32; j++) {
            // Simple copy for now - should be replaced with optimized version
            dst[j * dst_stride + i] = src[i * src_stride + j];
        }
    }
}

void xdnn_transpose16x32_packBA16a16b2a_v2(const XDNN_BF16 *src, int src_stride, XDNN_BF16 *dst, int dst_stride) {
    // Alternative implementation of specialized transpose
    // For now, same as v1
    xdnn_transpose16x32_packBA16a16b2a_v1(src, src_stride, dst, dst_stride);
}

void xdnn_transpose16xN_packBA16a16b2a_v1(const XDNN_BF16 *src, int src_cols, int src_stride, XDNN_BF16 *dst, int dst_rows, int dst_stride) {
    // 16xN specialized transpose for BF16
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < src_cols; j++) {
            // Simple copy for now - should be replaced with optimized version
            dst[j * dst_stride + i] = src[i * src_stride + j];
        }
    }
}

// Special packing transposes for FP16
void xdnn_transpose16x32_packBA16a16b2a_v1(const XDNN_FP16 *src, int src_stride, XDNN_FP16 *dst, int dst_stride) {
    // Specialized transpose for FP16 with packing pattern
    // This needs to be implemented with proper optimizations later
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 32; j++) {
            // Simple copy for now - should be replaced with optimized version
            dst[j * dst_stride + i] = src[i * src_stride + j];
        }
    }
}

void xdnn_transpose16x32_packBA16a16b2a_v2(const XDNN_FP16 *src, int src_stride, XDNN_FP16 *dst, int dst_stride) {
    // Alternative implementation of specialized transpose
    // For now, same as v1
    xdnn_transpose16x32_packBA16a16b2a_v1(src, src_stride, dst, dst_stride);
}

void xdnn_transpose16xN_packBA16a16b2a_v1(const XDNN_FP16 *src, int src_cols, int src_stride, XDNN_FP16 *dst, int dst_rows, int dst_stride) {
    // 16xN specialized transpose for FP16
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < src_cols; j++) {
            // Simple copy for now - should be replaced with optimized version
            dst[j * dst_stride + i] = src[i * src_stride + j];
        }
    }
}
