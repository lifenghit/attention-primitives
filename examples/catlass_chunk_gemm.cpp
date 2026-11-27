#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#define ATTENTION_ENABLE_CATLASS_DEVICE_ADAPTER
#include "attention/catlass_adapter.hpp"

namespace {

constexpr uint32_t kL = 128;
constexpr uint32_t kN = 128;
constexpr size_t kInputElements = static_cast<size_t>(kL) * kN;
constexpr size_t kInputBytes = kInputElements * sizeof(uint16_t);
constexpr size_t kOutputBytes = kInputElements * sizeof(float);

#define ACL_CHECK(expression)                                                                  \
    do {                                                                                       \
        const aclError status = (expression);                                                   \
        if (status != ACL_SUCCESS) {                                                            \
            std::cerr << #expression << " failed with ACL error " << status << " at line "     \
                      << __LINE__ << std::endl;                                                 \
            return false;                                                                       \
        }                                                                                      \
    } while (false)

uint16_t FloatToFp16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000U;
    int exponent = static_cast<int>((bits >> 23) & 0xffU) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t result = mantissa >> shift;
        if (((mantissa >> (shift - 1)) & 1U) != 0U) {
            ++result;
        }
        return static_cast<uint16_t>(sign | result);
    }
    uint32_t result = (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
    if ((mantissa & 0x1000U) != 0U) {
        ++result;
    }
    return static_cast<uint16_t>(sign | result);
}

float Fp16ToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fU;
    const uint32_t mantissa = value & 0x3ffU;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int adjustedExponent = 127 - 15 + 1;
            uint32_t adjustedMantissa = mantissa;
            while ((adjustedMantissa & 0x400U) == 0) {
                adjustedMantissa <<= 1;
                --adjustedExponent;
            }
            adjustedMantissa &= 0x3ffU;
            bits = sign | (static_cast<uint32_t>(adjustedExponent) << 23) | (adjustedMantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

} // namespace

__global__ __aicore__ void CatlassChunkBCGemmKernel(GM_ADDR c, GM_ADDR b, GM_ADDR m)
{
    const attention::catlass::BCGemm::Arguments arguments{
        Catlass::GemmCoord(kL, kL, kN), c, b, m};
    attention::catlass::BCGemm::Run(arguments);
}

namespace {

bool Run()
{
    aclrtStream stream = nullptr;
    uint8_t* deviceB = nullptr;
    uint8_t* deviceC = nullptr;
    uint8_t* deviceM = nullptr;

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    ACL_CHECK(aclrtCreateStream(&stream));

    std::vector<uint16_t> hostB(kInputElements);
    std::vector<uint16_t> hostC(kInputElements);
    for (uint32_t n = 0; n < kN; ++n) {
        for (uint32_t l = 0; l < kL; ++l) {
            const size_t offset = static_cast<size_t>(n) * kL + l;
            const float b = static_cast<float>(static_cast<int>((n * 3 + l * 5) % 17) - 8) / 64.0F;
            const float c = static_cast<float>(static_cast<int>((n * 7 + l * 11) % 19) - 9) / 64.0F;
            hostB[offset] = FloatToFp16(b);
            hostC[offset] = FloatToFp16(c);
        }
    }

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), kInputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), kInputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceM), kOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, kInputBytes, hostB.data(), kInputBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceC, kInputBytes, hostC.data(), kInputBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(deviceM, kOutputBytes, 0, kOutputBytes));

    CatlassChunkBCGemmKernel<<<1, nullptr, stream>>>(deviceC, deviceB, deviceM);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<float> hostM(kInputElements);
    ACL_CHECK(aclrtMemcpy(hostM.data(), kOutputBytes, deviceM, kOutputBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    float maxAbsoluteError = 0.0F;
    size_t worstIndex = 0;
    float worstExpected = 0.0F;
    size_t failures = 0;
    for (uint32_t i = 0; i < kL; ++i) {
        for (uint32_t j = 0; j < kL; ++j) {
            float expected = 0.0F;
            for (uint32_t n = 0; n < kN; ++n) {
                expected += Fp16ToFloat(hostC[static_cast<size_t>(n) * kL + i]) *
                            Fp16ToFloat(hostB[static_cast<size_t>(n) * kL + j]);
            }
            const size_t index = static_cast<size_t>(i) * kL + j;
            const float error = std::abs(hostM[index] - expected);
            if (index == 0 || error > maxAbsoluteError) {
                maxAbsoluteError = error;
                worstIndex = index;
                worstExpected = expected;
            }
            if (error > 0.002F + 0.002F * std::abs(expected)) {
                ++failures;
            }
        }
    }

    std::cout << "BC shape: M=" << kL << ", N=" << kL << ", K=" << kN << std::endl;
    std::cout << "max_absolute_error=" << maxAbsoluteError << ", mismatches=" << failures << std::endl;
    std::cout << "worst_index=" << worstIndex << ", expected=" << worstExpected
              << ", actual=" << hostM[worstIndex] << std::endl;
    for (size_t i = 0; i < 4; ++i) {
        std::cout << "M[" << i << "]=" << hostM[i] << (i == 3 ? '\n' : ' ');
    }

    ACL_CHECK(aclrtFree(deviceM));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return failures == 0;
}

} // namespace

int main()
{
    const bool passed = Run();
    std::cout << (passed ? "CATLASS BC GEMM PASS" : "CATLASS BC GEMM FAIL") << std::endl;
    return passed ? 0 : 1;
}
