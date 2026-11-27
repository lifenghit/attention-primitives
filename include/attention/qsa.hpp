#ifndef ATTENTION_QSA_HPP_
#define ATTENTION_QSA_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace attention {

enum class QSAStatus {
  kOk,
  kInvalidArgument,
  kPhysicalPageOutOfBounds,
};

inline QSAStatus qsa_visible_blocks(std::int32_t position,
                                    std::int32_t sequence_length,
                                    std::int32_t compress_ratio,
                                    std::int32_t* visible) {
  if (visible == nullptr || position < 0 || sequence_length < 0 ||
      position >= sequence_length || compress_ratio <= 0) {
    return QSAStatus::kInvalidArgument;
  }
  *visible = std::min((position + 1) / compress_ratio,
                      sequence_length / compress_ratio);
  return QSAStatus::kOk;
}

struct QSACompressParams {
  const float* input = nullptr;
  float* output = nullptr;
  std::int32_t sequence_length = 0;
  std::int32_t heads = 0;
  std::int32_t head_dim = 0;
  std::int32_t compress_ratio = 0;
};

inline QSAStatus qsa_compress_groups(const QSACompressParams& params) {
  if (params.sequence_length < 0 || params.heads <= 0 ||
      params.head_dim <= 0 || params.compress_ratio <= 0 ||
      (params.sequence_length > 0 &&
       (params.input == nullptr || params.output == nullptr))) {
    return QSAStatus::kInvalidArgument;
  }
  const std::size_t heads = static_cast<std::size_t>(params.heads);
  const std::size_t dim = static_cast<std::size_t>(params.head_dim);
  const std::size_t ratio = static_cast<std::size_t>(params.compress_ratio);
  const std::size_t blocks = static_cast<std::size_t>(params.sequence_length) / ratio;
  for (std::size_t block = 0; block < blocks; ++block) {
    for (std::size_t head = 0; head < heads; ++head) {
      for (std::size_t axis = 0; axis < dim; ++axis) {
        float sum = 0.0F;
        for (std::size_t offset = 0; offset < ratio; ++offset) {
          const std::size_t token = block * ratio + offset;
          sum += params.input[(token * heads + head) * dim + axis];
        }
        params.output[(block * heads + head) * dim + axis] =
            sum / params.compress_ratio;
      }
    }
  }
  return QSAStatus::kOk;
}

struct QSAScoreParams {
  const float* index_q = nullptr;
  const float* compressed_k = nullptr;
  const std::int32_t* block_table = nullptr;
  float* scores = nullptr;
  std::int32_t* visible_blocks = nullptr;
  std::int32_t sequence_length = 0;
  std::int32_t index_heads = 0;
  std::int32_t head_dim = 0;
  std::int32_t logical_blocks = 0;
  std::int32_t physical_block_count = 0;
  std::int32_t compress_ratio = 0;
  float score_divisor = 0.0F;
};

inline QSAStatus qsa_scores(const QSAScoreParams& params) {
  if (params.sequence_length < 0 || params.index_heads <= 0 ||
      params.head_dim <= 0 || params.logical_blocks < 0 ||
      params.physical_block_count < 0 || params.compress_ratio <= 0 ||
      !std::isfinite(params.score_divisor) || params.score_divisor <= 0.0F ||
      params.logical_blocks != params.sequence_length / params.compress_ratio) {
    return QSAStatus::kInvalidArgument;
  }
  const std::size_t length = static_cast<std::size_t>(params.sequence_length);
  const std::size_t heads = static_cast<std::size_t>(params.index_heads);
  const std::size_t dim = static_cast<std::size_t>(params.head_dim);
  const std::size_t logical_blocks = static_cast<std::size_t>(params.logical_blocks);
  if (length > 0 && (params.index_q == nullptr || params.scores == nullptr ||
                     params.visible_blocks == nullptr)) {
    return QSAStatus::kInvalidArgument;
  }
  if (params.physical_block_count > 0 && params.compressed_k == nullptr) {
    return QSAStatus::kInvalidArgument;
  }
  if (params.block_table == nullptr &&
      params.physical_block_count != params.logical_blocks) {
    return QSAStatus::kInvalidArgument;
  }

  const float negative_infinity = -std::numeric_limits<float>::infinity();
  for (std::size_t position = 0; position < length; ++position) {
    std::int32_t visible = 0;
    const QSAStatus status = qsa_visible_blocks(
        static_cast<std::int32_t>(position), params.sequence_length,
        params.compress_ratio, &visible);
    if (status != QSAStatus::kOk) {
      return status;
    }
    params.visible_blocks[position] = visible;
    float* row = params.scores + position * logical_blocks;
    std::fill(row, row + logical_blocks, negative_infinity);
    for (std::int32_t logical = 0; logical < visible; ++logical) {
      const std::int32_t physical = params.block_table == nullptr
                                        ? logical
                                        : params.block_table[logical];
      if (physical < 0 || physical >= params.physical_block_count) {
        return QSAStatus::kPhysicalPageOutOfBounds;
      }
      const float* key = params.compressed_k +
                         static_cast<std::size_t>(physical) * dim;
      float score = 0.0F;
      for (std::size_t head = 0; head < heads; ++head) {
        const float* query = params.index_q + (position * heads + head) * dim;
        float dot = 0.0F;
        for (std::size_t axis = 0; axis < dim; ++axis) {
          dot += query[axis] * key[axis];
        }
        score += std::max(dot, 0.0F);
      }
      row[logical] = score / params.score_divisor;
    }
  }
  return QSAStatus::kOk;
}

struct QSAParams {
  const float* q = nullptr;
  const float* k = nullptr;
  const float* v = nullptr;
  const float* index_q = nullptr;
  const float* index_k = nullptr;
  const std::int32_t* block_table = nullptr;
  float* output = nullptr;
  float* scores = nullptr;
  std::int32_t* visible_blocks = nullptr;
  std::int32_t* selected_indices = nullptr;
  std::int32_t* physical_blocks = nullptr;
  std::int32_t* selected_counts = nullptr;
  std::int32_t sequence_length = 0;
  std::int32_t heads = 0;
  std::int32_t kv_heads = 0;
  std::int32_t index_heads = 0;
  std::int32_t head_dim = 0;
  std::int32_t value_dim = 0;
  std::int32_t index_dim = 0;
  std::int32_t physical_block_count = 0;
  std::int32_t compress_ratio = 0;
  std::int32_t top_k = 0;
  float score_divisor = 0.0F;
  float query_scale = 0.0F;
};

inline QSAStatus qsa(const QSAParams& params) {
  if (params.sequence_length < 0 || params.heads <= 0 ||
      params.kv_heads <= 0 || params.index_heads <= 0 ||
      params.head_dim <= 0 || params.value_dim <= 0 || params.index_dim <= 0 ||
      params.heads % params.kv_heads != 0 || params.compress_ratio <= 0 ||
      params.top_k <= 0 || !std::isfinite(params.query_scale) ||
      !std::isfinite(params.score_divisor) || params.score_divisor <= 0.0F ||
      params.physical_block_count < 0) {
    return QSAStatus::kInvalidArgument;
  }
  const std::size_t length = static_cast<std::size_t>(params.sequence_length);
  const std::size_t heads = static_cast<std::size_t>(params.heads);
  const std::size_t kv_heads = static_cast<std::size_t>(params.kv_heads);
  const std::size_t dim = static_cast<std::size_t>(params.head_dim);
  const std::size_t value_dim = static_cast<std::size_t>(params.value_dim);
  const std::size_t index_dim = static_cast<std::size_t>(params.index_dim);
  const std::size_t ratio = static_cast<std::size_t>(params.compress_ratio);
  const std::size_t logical = length / ratio;
  const std::size_t top_k = static_cast<std::size_t>(params.top_k);
  const std::size_t physical = params.block_table == nullptr
                                   ? logical
                                   : static_cast<std::size_t>(params.physical_block_count);
  if (length > 0 &&
      (params.q == nullptr || params.k == nullptr || params.v == nullptr ||
       params.index_q == nullptr || params.index_k == nullptr ||
       params.output == nullptr || params.scores == nullptr ||
       params.visible_blocks == nullptr || params.selected_indices == nullptr ||
       params.physical_blocks == nullptr || params.selected_counts == nullptr)) {
    return QSAStatus::kInvalidArgument;
  }

  std::vector<float> pooled_k(logical * kv_heads * dim, 0.0F);
  std::vector<float> pooled_v(logical * kv_heads * value_dim, 0.0F);
  std::vector<float> pooled_index(logical * index_dim, 0.0F);
  QSACompressParams compress;
  compress.sequence_length = params.sequence_length;
  compress.compress_ratio = params.compress_ratio;
  compress.input = params.k;
  compress.output = pooled_k.data();
  compress.heads = params.kv_heads;
  compress.head_dim = params.head_dim;
  QSAStatus status = qsa_compress_groups(compress);
  if (status != QSAStatus::kOk) {
    return status;
  }
  compress.input = params.v;
  compress.output = pooled_v.data();
  compress.head_dim = params.value_dim;
  status = qsa_compress_groups(compress);
  if (status != QSAStatus::kOk) {
    return status;
  }
  compress.input = params.index_k;
  compress.output = pooled_index.data();
  compress.heads = 1;
  compress.head_dim = params.index_dim;
  status = qsa_compress_groups(compress);
  if (status != QSAStatus::kOk) {
    return status;
  }

  // The top-level reference owns an identity physical cache. A block table can
  // permute those slots, but cannot address beyond the generated cache.
  if (physical > logical) {
    return QSAStatus::kPhysicalPageOutOfBounds;
  }
  QSAScoreParams score_params;
  score_params.index_q = params.index_q;
  score_params.compressed_k = pooled_index.data();
  score_params.block_table = params.block_table;
  score_params.scores = params.scores;
  score_params.visible_blocks = params.visible_blocks;
  score_params.sequence_length = params.sequence_length;
  score_params.index_heads = params.index_heads;
  score_params.head_dim = params.index_dim;
  score_params.logical_blocks = static_cast<std::int32_t>(logical);
  score_params.physical_block_count = static_cast<std::int32_t>(physical);
  score_params.compress_ratio = params.compress_ratio;
  score_params.score_divisor = params.score_divisor;
  status = qsa_scores(score_params);
  if (status != QSAStatus::kOk) {
    return status;
  }

  const std::size_t group_size = heads / kv_heads;
  std::vector<std::int32_t> order(logical);
  std::vector<float> weights(top_k);
  for (std::size_t position = 0; position < length; ++position) {
    const std::size_t count = std::min(
        top_k, static_cast<std::size_t>(params.visible_blocks[position]));
    const float* score_row = params.scores + position * logical;
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [score_row](std::int32_t lhs,
                                                       std::int32_t rhs) {
      return score_row[lhs] > score_row[rhs] ||
             (score_row[lhs] == score_row[rhs] && lhs < rhs);
    });
    params.selected_counts[position] = static_cast<std::int32_t>(count);
    for (std::size_t slot = 0; slot < top_k; ++slot) {
      const std::size_t offset = position * top_k + slot;
      if (slot < count) {
        const std::int32_t logical_block = order[slot];
        params.selected_indices[offset] = logical_block;
        params.physical_blocks[offset] = params.block_table == nullptr
                                             ? logical_block
                                             : params.block_table[logical_block];
      } else {
        params.selected_indices[offset] = -1;
        params.physical_blocks[offset] = -1;
      }
    }

    for (std::size_t head = 0; head < heads; ++head) {
      float* out = params.output + (position * heads + head) * value_dim;
      std::fill(out, out + value_dim, 0.0F);
      if (count == 0) {
        continue;
      }
      const std::size_t kv_head = head / group_size;
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t slot = 0; slot < count; ++slot) {
        const std::int32_t block = params.physical_blocks[position * top_k + slot];
        const float* query = params.q + (position * heads + head) * dim;
        const float* key = pooled_k.data() +
                           (static_cast<std::size_t>(block) * kv_heads + kv_head) * dim;
        float dot = 0.0F;
        for (std::size_t axis = 0; axis < dim; ++axis) {
          dot += query[axis] * key[axis];
        }
        weights[slot] = dot * params.query_scale;
        maximum = std::max(maximum, weights[slot]);
      }
      float denominator = 0.0F;
      for (std::size_t slot = 0; slot < count; ++slot) {
        weights[slot] = std::exp(weights[slot] - maximum);
        denominator += weights[slot];
      }
      for (std::size_t slot = 0; slot < count; ++slot) {
        const std::int32_t block = params.physical_blocks[position * top_k + slot];
        const float* value = pooled_v.data() +
                             (static_cast<std::size_t>(block) * kv_heads + kv_head) * value_dim;
        for (std::size_t axis = 0; axis < value_dim; ++axis) {
          out[axis] += weights[slot] * value[axis] / denominator;
        }
      }
    }
  }
  return QSAStatus::kOk;
}

}  // namespace attention

#endif  // ATTENTION_QSA_HPP_
