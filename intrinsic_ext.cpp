#include "conversion.h"
#include "intrinsic_ext.h"
#include "conversion.h"
#include "debug_print.h"

// Implementation of Load BF16 and Convert BF16 to FP32 functions
__m512 _mm512_loadu_pbh(void const *mem_addr) {
    DEBUG_PRINT();
    return _mm512_setzero_ps();
}

// AVX512 implementation for masked load
__m512 _mm512_maskz_loadu_pbh(__mmask16 k, void const *mem_addr) {
    DEBUG_PRINT();
    return _mm512_setzero_ps();
}

// AVX512 implementation for store
void _mm512_storeu_pbh(void *mem_addr, __m512 a) {
    DEBUG_PRINT();
}

void _mm512_mask_storeu_pbh(void *mem_addr, __mmask16 k, __m512 a) {
    DEBUG_PRINT();
}
