#include "conversion.h"
#include "intrinsic_ext.h"
#include "conversion.h"
#include "debug_print.h"

// Implementation of Load BF16 and Convert BF16 to FP32 functions
__m512 _mm512_loadu_pbh(void const *mem_addr) {
    DEBUG_PRINT();
    // Load 16 BF16 values (32 bytes) into a 256-bit register
    __m256i bf16_data = _mm256_loadu_si256((__m256i const*)mem_addr);
    // First, we'll convert to 32-bit integers (zero-extended)
    __m512i int32_data = _mm512_cvtepu16_epi32(bf16_data);
    // Shift left by 16 bits to position the BF16 bits correctly in FP32 format
    __m512i shifted_data = _mm512_slli_epi32(int32_data, 16);
    // Reinterpret as FP32 values
    return _mm512_castsi512_ps(shifted_data);
}

// AVX512 implementation for masked load
__m512 _mm512_maskz_loadu_pbh(__mmask16 k, void const *mem_addr) {
    DEBUG_PRINT();
    // Similar to _mm512_loadu_pbh but with a mask
    __m256i bf16_data = _mm256_loadu_si256((__m256i const*)mem_addr);
    // Convert BF16 to FP32 by shifting left to 16 bits
    __m512i int32_data = _mm512_cvtepu16_epi32(bf16_data);
    __m512i shifted_data = _mm512_slli_epi32(int32_data, 16);
    // Apply mask - zeroing out positions where k bit is 0
    return _mm512_maskz_mov_ps(k, _mm512_castsi512_ps(shifted_data));
}

// AVX512 implementation for store
void _mm512_storeu_pbh(void *mem_addr, __m512 a) {
    DEBUG_PRINT();
    // Convert FP32 to 32-bit integers
    __m512i int_val = _mm512_castps_si512(a);
    // Round the mantissa (add 0x7FFF for rounding)
    __m512i rounding_bias = _mm512_set1_epi32(0x7FFF);
    __m512i rounded_val = _mm512_add_epi32(int_val, rounding_bias);
    // Shift right by 16 bits to get the BF16 representation
    __m512i bf16_val = _mm512_srli_epi32(rounded_val, 16);
    // Pack the 16 32-bit values into 16 16-bit values
    __m256i packed_bf16 = _mm512_cvtepi32_epi16(bf16_val);
    // Store the BF16 values back to memory
    _mm256_storeu_si256((__m256i*)mem_addr, packed_bf16);
}

void _mm512_mask_storeu_pbh(void *mem_addr, __mmask16 k, __m512 a) {
    DEBUG_PRINT();
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
