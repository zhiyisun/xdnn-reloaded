#include "conversion.h"
#include "intrinsic_ext.h"
#include "conversion.h"

// Fallback implementation for systems without AVX512 support
#ifndef __AVX512F__

// Implementation of Load BF16 and Convert BF16 to FP32 functions using 
// standard code instead of AVX512 intrinsics for compatibility
__m512 _mm512_loadu_pbh(void const *mem_addr) {
    // Create a fallback implementation
    alignas(64) float result[16];
    
    // Convert individual BF16 values to FP32
    const uint16_t* bf16_ptr = static_cast<const uint16_t*>(mem_addr);
    
    for (int i = 0; i < 16; i++) {
        // Convert the BF16 representation to FP32
        uint32_t bf16_bits = static_cast<uint32_t>(bf16_ptr[i]);
        uint32_t fp32_bits = bf16_bits << 16;
        float* fp32_ptr = reinterpret_cast<float*>(&fp32_bits);
        result[i] = *fp32_ptr;
    }
    
    // Load the data into m512 manually
    return _mm512_loadu_ps(result);
}

#else
// Implementation of Load BF16 and Convert BF16 to FP32 functions
__m512 _mm512_loadu_pbh(void const *mem_addr) {
    // Load 16 BF16 values (32 bytes) into a 256-bit register
    __m256i bf16_data = _mm256_loadu_si256((__m256i const*)mem_addr);
    
    // First, we'll convert to 32-bit integers (zero-extended)
    __m512i int32_data = _mm512_cvtepu16_epi32(bf16_data);
    
    // Shift left by 16 bits to position the BF16 bits correctly in FP32 format
    __m512i shifted_data = _mm512_slli_epi32(int32_data, 16);
    
    // Reinterpret as FP32 values
    return _mm512_castsi512_ps(shifted_data);
}
#endif

#ifndef __AVX512F__
// Fallback implementation for masked load
__m512 _mm512_maskz_loadu_pbh(__mmask16 k, void const *mem_addr) {
    // Create a fallback implementation
    alignas(64) float result[16] = {0}; // Initialize to zero for masked elements
    
    // Convert individual BF16 values to FP32 where mask bits are set
    const uint16_t* bf16_ptr = static_cast<const uint16_t*>(mem_addr);
    
    for (int i = 0; i < 16; i++) {
        if ((k >> i) & 1) {  // Check if this bit is set in the mask
            // Convert the BF16 representation to FP32
            uint32_t bf16_bits = static_cast<uint32_t>(bf16_ptr[i]);
            uint32_t fp32_bits = bf16_bits << 16;
            float* fp32_ptr = reinterpret_cast<float*>(&fp32_bits);
            result[i] = *fp32_ptr;
        }
    }
    
    // Load the data into m512 manually
    return _mm512_loadu_ps(result);
}

// Fallback implementation for store
void _mm512_storeu_pbh(void *mem_addr, __m512 a) {
    alignas(64) float temp[16];
    _mm512_storeu_ps(temp, a);
    
    uint16_t* bf16_ptr = static_cast<uint16_t*>(mem_addr);
    
    for (int i = 0; i < 16; i++) {
        // Extract the upper 16 bits of the FP32 value
        uint32_t* fp32_bits = reinterpret_cast<uint32_t*>(&temp[i]);
        // Round the mantissa (simplified rounding)
        *fp32_bits += 0x7FFF;
        // Store the upper 16 bits as BF16
        bf16_ptr[i] = static_cast<uint16_t>(*fp32_bits >> 16);
    }
}

#else
// AVX512 implementation for masked load
__m512 _mm512_maskz_loadu_pbh(__mmask16 k, void const *mem_addr) {
    // Similar to _mm512_loadu_pbh but with a mask
    __m256i bf16_data = _mm256_loadu_si256((__m256i const*)mem_addr);
    
    // Convert BF16 to FP32 by shifting left by 16 bits
    __m512i int32_data = _mm512_cvtepu16_epi32(bf16_data);
    __m512i shifted_data = _mm512_slli_epi32(int32_data, 16);
    
    // Apply mask - zeroing out positions where k bit is 0
    return _mm512_maskz_mov_ps(k, _mm512_castsi512_ps(shifted_data));
}

// AVX512 implementation for store
void _mm512_storeu_pbh(void *mem_addr, __m512 a) {
    // Convert FP32 to BF16 by rounding and truncating
    
    // 1. Convert FP32 to 32-bit integers
    __m512i int_val = _mm512_castps_si512(a);
    
    // 2. Round the mantissa (add 0x7FFF for rounding)
    __m512i rounding_bias = _mm512_set1_epi32(0x7FFF);
    __m512i rounded_val = _mm512_add_epi32(int_val, rounding_bias);
    
    // 3. Shift right by 16 bits to get the BF16 representation
        __m512i bf16_val = _mm512_srli_epi32(rounded_val, 16);
#endif
    
#ifdef __AVX512F__
    // 4. Pack the 16 32-bit values into 16 16-bit values
    __m512i permute_mask = _mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i permuted = _mm512_permutexvar_epi32(permute_mask, bf16_val);
    __m256i bf16_result = _mm512_cvtepi32_epi16(permuted);
    
    // 5. Store the BF16 values back to memory
    _mm256_storeu_si256((__m256i*)mem_addr, bf16_result);
#else
    __m256i packed_bf16 = _mm256_set1_epi32(0); // Fallback for non-AVX512 systems
    
    // 5. Store the packed BF16 values to memory
    _mm256_storeu_si256((__m256i*)mem_addr, packed_bf16);
#endif
}

void _mm512_mask_storeu_pbh(void *mem_addr, __mmask16 k, __m512 a) {
    // Similar to _mm512_storeu_pbh but with a mask
    
    // First, load existing values to merge with
    __m256i existing = _mm256_loadu_si256((__m256i*)mem_addr);
    
    // Convert FP32 to 32-bit integers
    __m512i int_val = _mm512_castps_si512(a);
    
    // Round the mantissa
    __m512i rounding_bias = _mm512_set1_epi32(0x7FFF);
    __m512i rounded_val = _mm512_add_epi32(int_val, rounding_bias);
    
    // Shift right by 16 bits
    __m512i bf16_val = _mm512_srli_epi32(rounded_val, 16);
    
    // Pack the 16 32-bit values into 16 16-bit values
    __m256i packed_bf16 = _mm512_cvtepi32_epi16(bf16_val);
    
    // Apply mask: combine existing values with new values based on mask
    __m256i masked_result = _mm256_mask_mov_epi16(existing, k, packed_bf16);
    
    // Store the result
    _mm256_storeu_si256((__m256i*)mem_addr, masked_result);
}
