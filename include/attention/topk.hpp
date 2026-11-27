#ifndef ATTENTION_TOPK_HPP_
#define ATTENTION_TOPK_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace attention {

enum class TopKStatus {
  kOk,
  kInvalidArgument,
};

struct TopKParams {
  const float* scores = nullptr;
  std::int32_t* indices = nullptr;
  std::int32_t* counts = nullptr;
  std::int32_t q_blocks = 0;
  std::int32_t q_heads = 0;
  std::int32_t kv_blocks = 0;
  std::int32_t top_k = 0;
};

inline TopKStatus topk(const TopKParams& params) {
  if (params.q_blocks < 0 || params.q_heads <= 0 || params.kv_blocks < 0 ||
      params.top_k <= 0) {
    return TopKStatus::kInvalidArgument;
  }
  const std::size_t rows = static_cast<std::size_t>(params.q_blocks) *
                           static_cast<std::size_t>(params.q_heads);
  const std::size_t kv_blocks = static_cast<std::size_t>(params.kv_blocks);
  const std::size_t top_k = static_cast<std::size_t>(params.top_k);
  if ((rows * kv_blocks > 0 && params.scores == nullptr) ||
      (rows > 0 && (params.indices == nullptr || params.counts == nullptr))) {
    return TopKStatus::kInvalidArgument;
  }

  std::vector<std::int32_t> order(kv_blocks);
  for (std::size_t row = 0; row < rows; ++row) {
    const float* scores = kv_blocks == 0 ? nullptr : params.scores + row * kv_blocks;
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [scores](std::int32_t lhs,
                                                   std::int32_t rhs) {
      const float lhs_score = scores[lhs];
      const float rhs_score = scores[rhs];
      return lhs_score > rhs_score ||
             (lhs_score == rhs_score && lhs < rhs);
    });

    std::int32_t count = 0;
    std::int32_t* indices = params.indices + row * top_k;
    const std::size_t limit = std::min(top_k, kv_blocks);
    for (std::size_t slot = 0; slot < limit; ++slot) {
      const std::int32_t index = order[slot];
      if (scores[index] == -INFINITY) {
        break;
      }
      indices[slot] = index;
      ++count;
    }
    for (std::size_t slot = static_cast<std::size_t>(count); slot < top_k;
         ++slot) {
      indices[slot] = -1;
    }
    params.counts[row] = count;
  }
  return TopKStatus::kOk;
}

}  // namespace attention

#endif  // ATTENTION_TOPK_HPP_
