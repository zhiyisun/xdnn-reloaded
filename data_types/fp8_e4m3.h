#pragma once
#include <cstdint>
#include <cmath>

class XDNN_E4M3 {
   private:
    uint8_t val;

   public:
    XDNN_E4M3() { val = 0; }
    XDNN_E4M3(float v) { (*this) = val; }
    XDNN_E4M3(uint8_t v) { val = v; }

    XDNN_E4M3 &operator=(uint8_t v) {
        val = v;
        return *this;
    }

    XDNN_E4M3 &operator=(const XDNN_E4M3 &v) {
        val = v.val;
        return *this;
    }

    XDNN_E4M3 &operator=(float value) {
        // 提取符号
        uint8_t sign = (value < 0) ? 1 : 0;
        if (sign) value = -value;

        // 提取指数和尾数
        int exponent;
        float mantissa = std::frexp(value, &exponent);

        // 调整指数以适应 e4m3 格式
        exponent += 7; // 偏移量为 7
        if (exponent < 0) exponent = 0;
        if (exponent > 15) exponent = 15;

        // 调整尾数以适应 3 位
        mantissa = std::ldexp(mantissa, 3); // 左移 3 位
        uint8_t mantissa_bits = static_cast<uint8_t>(mantissa) & 0x07; // 取低 3 位

        // 组合符号、指数和尾数
        val = (sign << 7) | (exponent << 3) | mantissa_bits;
        return *this;
    }   

    operator float() const {
        // Extract sign, exponent, and mantissa
        uint8_t sign = (val & 0x80) >> 7;
        uint8_t exponent = (val & 0x78) >> 3;
        uint8_t mantissa = val & 0x07;

        float result;
        if (exponent == 0) {
            // Subnormal: use (mantissa/8) * 2^(1-bias) with bias = 7 (i.e. 2^-6)
            result = (mantissa / 8.0f) * powf(2.0f, -6);
        } else {
            // Normalized: implicit 1, so (1 + mantissa/8) * 2^(exponent-bias)
            result = (1.0f + mantissa / 8.0f) * powf(2.0f, exponent - 7);
        }
        return sign ? -result : result;
    }

    // Conversion operator to bfloat16 (represented as uint16_t)
    operator uint16_t() const {
        // Convert fp8_e4m3 to float first.
        float f = static_cast<float>(*this);
        uint32_t bits = *(uint32_t *)&f;

        // The bfloat16 representation is the upper 16 bits of the float.
        uint16_t bf16 = static_cast<uint16_t>(bits >> 16);
        return bf16;
    }

        // Conversion operator to bfloat16 (represented as uint16_t)
        operator uint8_t() const {
            return val;
        }

} __attribute__((packed));
