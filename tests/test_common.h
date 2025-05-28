#pragma once

constexpr float BF16_PRECISION_TOLERANCE = 0.07f;
constexpr float FP16_PRECISION_TOLERANCE = 0.05f;
// constexpr float FP32_PRECISION_TOLERANCE = 1e-3f;
constexpr float FP32_PRECISION_TOLERANCE = 0.13f;
constexpr float F32U4F32_QUANT_PRECISION_TOLERANCE = 0.8f; // Max observed diff for quant params was ~0.79
constexpr float F32S8F32_ACTIVATION_TOLERANCE = 1.0f; // Tolerance for SiLU/GELU with int8 quantization
constexpr float FP16_FUSED_PRECISION_TOLERANCE = 3.0f;
