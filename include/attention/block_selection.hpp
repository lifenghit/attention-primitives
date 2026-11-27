#ifndef ATTENTION_BLOCK_SELECTION_HPP_
#define ATTENTION_BLOCK_SELECTION_HPP_

#include <cstddef>
#include <cstdint>

namespace attention {

enum class BlockSelectionStatus {
  kOk,
  kInvalidArgument,
  kIndexOutOfBounds,
};

struct BlockSelectionParams {
  const std::int32_t* topk_indices = nullptr;
  const std::int32_t* counts = nullptr;
  const std::int32_t* block_table = nullptr;
  std::int32_t* physical_blocks = nullptr;
  std::int32_t* dense_positions = nullptr;
  std::int32_t* output_counts = nullptr;
  std::int32_t q_blocks = 0;
  std::int32_t q_heads = 0;
  std::int32_t kv_blocks = 0;
  std::int32_t top_k = 0;
  std::int32_t physical_block_count = 0;
};

inline BlockSelectionStatus block_selection(const BlockSelectionParams& params) {
  if (params.q_blocks < 0 || params.q_heads <= 0 || params.kv_blocks < 0 ||
      params.top_k <= 0 || params.physical_block_count < 0) {
    return BlockSelectionStatus::kInvalidArgument;
  }
  const std::size_t rows = static_cast<std::size_t>(params.q_blocks) *
                           static_cast<std::size_t>(params.q_heads);
  const std::size_t top_k = static_cast<std::size_t>(params.top_k);
  if (rows > 0 && (params.topk_indices == nullptr || params.counts == nullptr ||
                   params.physical_blocks == nullptr ||
                   params.dense_positions == nullptr ||
                   params.output_counts == nullptr)) {
    return BlockSelectionStatus::kInvalidArgument;
  }
  if (params.block_table != nullptr && params.kv_blocks > 0 &&
      params.physical_block_count == 0) {
    return BlockSelectionStatus::kInvalidArgument;
  }

  for (std::size_t row = 0; row < rows; ++row) {
    const std::int32_t count = params.counts[row];
    if (count < 0 || static_cast<std::size_t>(count) > top_k ||
        count > params.kv_blocks) {
      return BlockSelectionStatus::kInvalidArgument;
    }
    for (std::int32_t slot = 0; slot < count; ++slot) {
      const std::int32_t logical = params.topk_indices[row * top_k +
                                                       static_cast<std::size_t>(slot)];
      if (logical < 0 || logical >= params.kv_blocks) {
        return BlockSelectionStatus::kIndexOutOfBounds;
      }
      if (params.block_table != nullptr) {
        const std::int32_t physical = params.block_table[logical];
        if (physical < 0 || physical >= params.physical_block_count) {
          return BlockSelectionStatus::kIndexOutOfBounds;
        }
      }
    }
  }

  for (std::size_t row = 0; row < rows; ++row) {
    const std::int32_t count = params.counts[row];
    params.output_counts[row] = count;
    for (std::size_t slot = 0; slot < top_k; ++slot) {
      if (slot < static_cast<std::size_t>(count)) {
        const std::int32_t logical = params.topk_indices[row * top_k + slot];
        params.physical_blocks[row * top_k + slot] =
            params.block_table == nullptr ? logical : params.block_table[logical];
        params.dense_positions[row * top_k + slot] = static_cast<std::int32_t>(slot);
      } else {
        params.physical_blocks[row * top_k + slot] = -1;
        params.dense_positions[row * top_k + slot] = -1;
      }
    }
  }
  return BlockSelectionStatus::kOk;
}

}  // namespace attention

#endif  // ATTENTION_BLOCK_SELECTION_HPP_
