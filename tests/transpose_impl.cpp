#include "transpose.h"
#include "data_types/float16.h"
#include "data_types/bfloat16.h"

// NOTE: This file is deprecated and can be removed.
// The real implementations are now available in /home/zhiyis/workspace/code/xdnn-reloaded/transpose_impl.cpp
// These are test-only implementations that were used when the library ones weren't linked properly.

// These implementations are intentionally left empty as they should not be used anymore.
// The compiler will prefer the real implementations from the main library.

/*
void xdnn_transpose(const float *src, int src_rows, int src_cols, int src_stride, float *dst, int dst_stride) {
    // Deprecated - use main library implementation
}

void xdnn_transpose(const XDNN_BF16 *src, int src_rows, int src_cols, int src_stride, XDNN_BF16 *dst, int dst_stride) {
    // Deprecated - use main library implementation
}

void xdnn_transpose(const int *src, int src_rows, int src_cols, int src_stride, int *dst, int dst_stride) {
    // Deprecated - use main library implementation
}
*/
