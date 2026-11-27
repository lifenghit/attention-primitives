#ifndef ATTENTION_BLOCK_SCORE_HPP_
#define ATTENTION_BLOCK_SCORE_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace attention {

struct Float16 {
  std::uint16_t bits;
};

inline float fp16_to_float(Float16 value) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(value.bits & 0x8000u) << 16;
  const std::uint32_t exponent = (value.bits >> 10) & 0x1fu;
  const std::uint32_t fraction = value.bits & 0x03ffu;
  std::uint32_t result;

  if (exponent == 0) {
    if (fraction == 0) {
      result = sign;
    } else {
      std::uint32_t normalized = fraction;
      int shift = 0;
      while ((normalized & 0x0400u) == 0) {
        normalized <<= 1;
        ++shift;
      }
      normalized &= 0x03ffu;
      result = sign | (static_cast<std::uint32_t>(113 - shift) << 23) |
               (normalized << 13);
    }
  } else if (exponent == 0x1fu) {
    result = sign | 0x7f800000u | (fraction << 13);
  } else {
    result = sign | ((exponent + 112) << 23) | (fraction << 13);
  }

  float output;
  std::memcpy(&output, &result, sizeof(output));
  return output;
}

enum class BlockScoreStatus {
  kOk,
  kInvalidArgument,
};

struct BlockScoreParams {
  const Float16* q = nullptr;
  const Float16* k = nullptr;
  float* scores = nullptr;
  std::int32_t q_blocks = 0;
  std::int32_t q_heads = 0;
  std::int32_t kv_blocks = 0;
  std::int32_t kv_heads = 0;
  std::int32_t d = 0;
  bool causal = false;
};

inline BlockScoreStatus block_score(const BlockScoreParams& params) {
  if (params.q_blocks < 0 || params.q_heads <= 0 || params.kv_blocks < 0 ||
      params.kv_heads <= 0 || params.d <= 0 ||
      params.q_heads % params.kv_heads != 0) {
    return BlockScoreStatus::kInvalidArgument;
  }
  const std::size_t rows = static_cast<std::size_t>(params.q_blocks) *
                           static_cast<std::size_t>(params.q_heads);
  const std::size_t score_count = rows * static_cast<std::size_t>(params.kv_blocks);
  if ((params.q_blocks > 0 && params.q == nullptr) ||
      (params.kv_blocks > 0 && params.k == nullptr) ||
      (score_count > 0 && params.scores == nullptr)) {
    return BlockScoreStatus::kInvalidArgument;
  }

  const std::size_t d = static_cast<std::size_t>(params.d);
  const std::size_t q_heads = static_cast<std::size_t>(params.q_heads);
  const std::size_t kv_blocks = static_cast<std::size_t>(params.kv_blocks);
  const std::size_t kv_heads = static_cast<std::size_t>(params.kv_heads);
  const std::size_t head_group = q_heads / kv_heads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(params.d));

  for (std::int32_t qb = 0; qb < params.q_blocks; ++qb) {
    for (std::int32_t qh = 0; qh < params.q_heads; ++qh) {
      const Float16* q = params.q +
                         (static_cast<std::size_t>(qb) * q_heads +
                          static_cast<std::size_t>(qh)) * d;
      float q_sum = 0.0F;
      for (std::size_t i = 0; i < d; ++i) {
        q_sum += fp16_to_float(q[i]);
      }
      const float q_mean = q_sum / static_cast<float>(params.d);
      const std::size_t kvh = static_cast<std::size_t>(qh) / head_group;

      for (std::int32_t kvb = 0; kvb < params.kv_blocks; ++kvb) {
        float& score = params.scores[
            (static_cast<std::size_t>(qb) * q_heads +
             static_cast<std::size_t>(qh)) * kv_blocks +
            static_cast<std::size_t>(kvb)];
        if (params.causal && kvb > qb) {
          score = -INFINITY;
          continue;
        }
        const Float16* k = params.k +
                           (static_cast<std::size_t>(kvb) * kv_heads + kvh) * d;
        float k_sum = 0.0F;
        for (std::size_t i = 0; i < d; ++i) {
          k_sum += fp16_to_float(k[i]);
        }
        score = q_mean * (k_sum / static_cast<float>(params.d)) * scale;
      }
    }
  }
  return BlockScoreStatus::kOk;
}

}  // namespace attention

#endif  // ATTENTION_BLOCK_SCORE_HPP_
