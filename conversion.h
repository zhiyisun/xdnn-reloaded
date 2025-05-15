#pragma once

#include "data_types/data_types.h"

// Conversion functions for different data types
inline float _xdnn_to_float(const XDNN_FP16& val) {
    return static_cast<float>(val);
}

inline float _xdnn_to_float(const XDNN_BF16& val) {
    return static_cast<float>(val);
}

inline float _xdnn_to_float(const XDNN_E4M3& val) {
    return static_cast<float>(val);
}

inline float _xdnn_to_float(float val) {
    return val;
}

inline XDNN_FP16 _xdnn_to_fp16(float val) {
    return XDNN_FP16(val);
}

inline XDNN_BF16 _xdnn_to_bf16(float val) {
    return XDNN_BF16(val);
}
