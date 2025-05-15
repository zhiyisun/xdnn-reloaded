#include "platform_detection.h"

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace xdnn {

// Helper function to execute cpuid instruction
void cpuid(int info[4], int infoType) {
#ifdef _WIN32
    __cpuid(info, infoType);
#else
    __cpuid(infoType, info[0], info[1], info[2], info[3]);
#endif
}

// Helper function to execute cpuid instruction with subleaf
void cpuidex(int info[4], int infoType, int subLeaf) {
#ifdef _WIN32
    __cpuidex(info, infoType, subLeaf);
#else
    __cpuid_count(infoType, subLeaf, info[0], info[1], info[2], info[3]);
#endif
}

// Implementation of CPU feature detection
CPUFeatures detectCPUFeatures() {
    CPUFeatures features;
    int info[4];
    
    // Check maximum supported CPUID leaf
    cpuid(info, 0);
    int maxLeaf = info[0];

    if (maxLeaf >= 1) {
        // Get features from EAX=1
        cpuid(info, 1);
        features.sse    = (info[3] & (1 << 25)) != 0;
        features.sse2   = (info[3] & (1 << 26)) != 0;
        features.sse3   = (info[2] & (1 << 0)) != 0;
        features.sse41  = (info[2] & (1 << 19)) != 0;
        features.sse42  = (info[2] & (1 << 20)) != 0;
        features.aesni  = (info[2] & (1 << 25)) != 0;
        features.avx    = (info[2] & (1 << 28)) != 0;
        features.fma    = (info[2] & (1 << 12)) != 0;
    }

    if (maxLeaf >= 7) {
        // Get features from EAX=7, ECX=0
        cpuidex(info, 7, 0);
        features.avx2    = (info[1] & (1 << 5)) != 0;
        features.avx512f = (info[1] & (1 << 16)) != 0;
        features.avx512dq = (info[1] & (1 << 17)) != 0;
        features.avx512bw = (info[1] & (1 << 30)) != 0;
        features.avx512vl = (info[1] & (1 << 31)) != 0;
    }

    return features;
}

// Get the best optimization level based on detected CPU features
OptimizationLevel getBestOptimizationLevel() {
    CPUFeatures features = detectCPUFeatures();
    
    if (features.supportsAVX512()) {
        return OptimizationLevel::AVX512;
    } else if (features.supportsAVX2()) {
        return OptimizationLevel::AVX2;
    } else if (features.supportsAVX()) {
        return OptimizationLevel::AVX;
    } else if (features.supportsSSE41()) {
        return OptimizationLevel::SSE41;
    } else if (features.supportsSSE2()) {
        return OptimizationLevel::SSE2;
    } else {
        return OptimizationLevel::Generic;
    }
}

// Check if a specific optimization level is supported at runtime
bool isOptimizationLevelSupported(OptimizationLevel level) {
    CPUFeatures features = detectCPUFeatures();
    
    switch (level) {
        case OptimizationLevel::Generic:
            return true;
        case OptimizationLevel::SSE2:
            return features.supportsSSE2();
        case OptimizationLevel::SSE41:
            return features.supportsSSE41();
        case OptimizationLevel::AVX:
            return features.supportsAVX();
        case OptimizationLevel::AVX2:
            return features.supportsAVX2();
        case OptimizationLevel::AVX512:
            return features.supportsAVX512();
        default:
            return false;
    }
}

} // namespace xdnn
