#pragma once

#include <cstdint>

namespace xdnn {

// CPU Feature detection
struct CPUFeatures {
    bool avx512f = false;    // AVX-512 Foundation
    bool avx512vl = false;   // AVX-512 Vector Length
    bool avx512bw = false;   // AVX-512 Byte and Word
    bool avx512dq = false;   // AVX-512 Doubleword and Quadword
    bool avx2 = false;       // AVX2
    bool avx = false;        // AVX
    bool sse42 = false;      // SSE4.2
    bool sse41 = false;      // SSE4.1
    bool sse3 = false;       // SSE3
    bool sse2 = false;       // SSE2
    bool sse = false;        // SSE
    bool fma = false;        // FMA
    bool aesni = false;      // AES-NI

    // Simple check for hardware capabilities needed by specific optimizations
    bool supportsAVX512() const { return avx512f && avx512vl && avx512bw && avx512dq; }
    bool supportsAVX2() const { return avx2; }
    bool supportsAVX() const { return avx; }
    bool supportsSSE42() const { return sse42; }
    bool supportsSSE41() const { return sse41; }
    bool supportsSSE3() const { return sse3; }
    bool supportsSSE2() const { return sse2; }
};

// Get CPU features through CPUID instructions
CPUFeatures detectCPUFeatures();

// Architecture-specific optimization levels
enum class OptimizationLevel {
    Generic,    // No architecture-specific optimizations
    SSE2,       // SSE2 optimizations
    SSE41,      // SSE4.1 optimizations
    AVX,        // AVX optimizations
    AVX2,       // AVX2 optimizations
    AVX512      // AVX-512 optimizations
};

// Determine the best optimization level based on detected CPU features
OptimizationLevel getBestOptimizationLevel();

// Check if a specific optimization level is supported at runtime
bool isOptimizationLevelSupported(OptimizationLevel level);

} // namespace xdnn
