#include "gtest/gtest.h"
#include "intrinsic_ext.h"
#include "data_types/bfloat16.h"
#include "test_common.h"
#include <vector>
#include <numeric>
#include <algorithm>

// Helper function to compare float arrays with a tolerance
void EXPECT_FLOAT_ARRAY_NEAR(const float* expected, const float* actual, size_t size, float tolerance) {
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(expected[i], actual[i], tolerance);
    }
}

// Test fixture for intrinsic_ext tests
class IntrinsicExtTest : public ::testing::Test {
protected:
    // Align memory to 64 bytes for AVX512 intrinsics
    void* aligned_alloc(size_t size, size_t alignment) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, size) != 0) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void aligned_free(void* ptr) {
        free(ptr);
    }

    std::vector<XDNN_BF16> bf16_data;
    std::vector<float> fp32_data;
    const size_t num_elements = AVX3_BF16_NUM; // 32 bf16 elements = 16 fp32 elements in __m512

    void SetUp() override {
        fp32_data.resize(num_elements);
        std::iota(fp32_data.begin(), fp32_data.end(), 1.0f); // Fill with 1.0, 2.0, ..., 32.0

        bf16_data.resize(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            bf16_data[i] = fp32_data[i];
        }
    }
};

// Test for _mm512_loadu_pbh
TEST_F(IntrinsicExtTest, LoaduPbh) {
    XDNN_BF16* bf16_ptr = static_cast<XDNN_BF16*>(aligned_alloc(num_elements * sizeof(XDNN_BF16), 64));
    std::copy(bf16_data.begin(), bf16_data.end(), bf16_ptr);

    __m512 loaded_data = _mm512_loadu_pbh(bf16_ptr);

    float result[AVX3_F32_NUM];
    _mm512_storeu_ps(result, loaded_data);

    // Compare the first 16 float values
    EXPECT_FLOAT_ARRAY_NEAR(fp32_data.data(), result, AVX3_F32_NUM, BF16_PRECISION_TOLERANCE); // BF16 has lower precision

    aligned_free(bf16_ptr);
}

// Test for _mm512_maskz_loadu_pbh
TEST_F(IntrinsicExtTest, MaskzLoaduPbh) {
    XDNN_BF16* bf16_ptr = static_cast<XDNN_BF16*>(aligned_alloc(num_elements * sizeof(XDNN_BF16), 64));
    std::copy(bf16_data.begin(), bf16_data.end(), bf16_ptr);

    __mmask16 k = 0xAAAA; // Example mask: 1010101010101010
    __m512 loaded_data = _mm512_maskz_loadu_pbh(k, bf16_ptr);

    float result[AVX3_F32_NUM];
    _mm512_storeu_ps(result, loaded_data);

    std::vector<float> expected_fp32(AVX3_F32_NUM);
    for (size_t i = 0; i < AVX3_F32_NUM; ++i) {
        if ((k >> i) & 1) {
            expected_fp32[i] = fp32_data[i];
        } else {
            expected_fp32[i] = 0.0f;
        }
    }
    EXPECT_FLOAT_ARRAY_NEAR(expected_fp32.data(), result, AVX3_F32_NUM, BF16_PRECISION_TOLERANCE);

    aligned_free(bf16_ptr);
}

// Test for _mm512_storeu_pbh
TEST_F(IntrinsicExtTest, StoreuPbh) {
    float source_fp32[AVX3_F32_NUM];
    std::iota(source_fp32, source_fp32 + AVX3_F32_NUM, 1.0f);
    __m512 fp32_vec = _mm512_loadu_ps(source_fp32);

    XDNN_BF16* bf16_storage = static_cast<XDNN_BF16*>(aligned_alloc(num_elements * sizeof(XDNN_BF16), 64));

    _mm512_storeu_pbh(bf16_storage, fp32_vec);

    std::vector<float> result_fp32(AVX3_F32_NUM);
    for (size_t i = 0; i < AVX3_F32_NUM; ++i) {
        result_fp32[i] = static_cast<float>(bf16_storage[i]);
    }

    EXPECT_FLOAT_ARRAY_NEAR(source_fp32, result_fp32.data(), AVX3_F32_NUM, BF16_PRECISION_TOLERANCE);

    aligned_free(bf16_storage);
}

// Test for _mm512_mask_storeu_pbh
TEST_F(IntrinsicExtTest, MaskStoreuPbh) {
    float source_fp32[AVX3_F32_NUM];
    std::iota(source_fp32, source_fp32 + AVX3_F32_NUM, 1.0f);
    __m512 fp32_vec = _mm512_loadu_ps(source_fp32);

    XDNN_BF16* bf16_storage = static_cast<XDNN_BF16*>(aligned_alloc(num_elements * sizeof(XDNN_BF16), 64));
    // Initialize with some other values to check if mask works
    std::vector<XDNN_BF16> initial_bf16(num_elements);
     for (size_t i = 0; i < num_elements; ++i) {
        initial_bf16[i] = 100.0f + i; // Different values
    }
    std::copy(initial_bf16.begin(), initial_bf16.end(), bf16_storage);


    __mmask16 k = 0x5555; // Example mask: 0101010101010101
    _mm512_mask_storeu_pbh(bf16_storage, k, fp32_vec);

    std::vector<float> result_fp32(AVX3_F32_NUM);
    std::vector<float> expected_fp32(AVX3_F32_NUM);

    for (size_t i = 0; i < AVX3_F32_NUM; ++i) {
        result_fp32[i] = static_cast<float>(bf16_storage[i]);
        if ((k >> i) & 1) {
            expected_fp32[i] = source_fp32[i];
        } else {
            expected_fp32[i] = static_cast<float>(initial_bf16[i]);
        }
    }

    EXPECT_FLOAT_ARRAY_NEAR(expected_fp32.data(), result_fp32.data(), AVX3_F32_NUM, BF16_PRECISION_TOLERANCE);

    aligned_free(bf16_storage);
}
