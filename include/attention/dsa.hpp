#ifndef ATTENTION_DSA_HPP_
#define ATTENTION_DSA_HPP_

#include "attention/topk.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace attention {

enum class DsaStatus {
  kOk,
  kInvalidArgument,
  kIndexOutOfBounds,
};

struct DsaIndexerParams {
  const float* q_idx = nullptr;
  const float* k_idx = nullptr;
  const float* w_idx = nullptr;
  std::int32_t* indices = nullptr;
  std::int32_t* counts = nullptr;
  std::int32_t batch = 0;
  std::int32_t sequence_length = 0;
  std::int32_t index_heads = 0;
  std::int32_t index_dim = 0;
  std::int32_t top_k = 0;
  float scale = std::numeric_limits<float>::quiet_NaN();
};

struct DsaParams {
  const float* q = nullptr;
  const float* k = nullptr;
  const float* v = nullptr;
  const float* q_idx = nullptr;
  const float* k_idx = nullptr;
  const float* w_idx = nullptr;
  const std::int32_t* indices = nullptr;
  float* output = nullptr;
  std::int32_t* selected_indices = nullptr;
  std::int32_t batch = 0;
  std::int32_t sequence_length = 0;
  std::int32_t query_heads = 0;
  std::int32_t kv_heads = 0;
  std::int32_t head_dim = 0;
  std::int32_t value_dim = 0;
  std::int32_t index_heads = 0;
  std::int32_t index_dim = 0;
  std::int32_t top_k = 0;
  float scale = std::numeric_limits<float>::quiet_NaN();
  float index_scale = std::numeric_limits<float>::quiet_NaN();
};

inline float dsa_default_scale(float scale, std::int32_t dim) {
  return std::isnan(scale) ? 1.0F / std::sqrt(static_cast<float>(dim)) : scale;
}

inline DsaStatus dsa_indexer(const DsaIndexerParams& params) {
  if (params.batch < 0 || params.sequence_length < 0 ||
      params.index_heads <= 0 || params.index_dim <= 0 || params.top_k <= 0) {
    return DsaStatus::kInvalidArgument;
  }
  const std::size_t batch = static_cast<std::size_t>(params.batch);
  const std::size_t length = static_cast<std::size_t>(params.sequence_length);
  const std::size_t heads = static_cast<std::size_t>(params.index_heads);
  const std::size_t dim = static_cast<std::size_t>(params.index_dim);
  const std::size_t top_k = static_cast<std::size_t>(params.top_k);
  const std::size_t token_count = batch * length;
  if (token_count > 0 &&
      (params.q_idx == nullptr || params.k_idx == nullptr ||
       params.indices == nullptr)) {
    return DsaStatus::kInvalidArgument;
  }
  const float scale = dsa_default_scale(params.scale, params.index_dim);
  std::vector<float> scores(length);
  std::vector<std::int32_t> row_counts(1);

  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t t = 0; t < length; ++t) {
      const std::size_t token = b * length + t;
      for (std::size_t s = 0; s < length; ++s) {
        if (s > t) {
          scores[s] = -std::numeric_limits<float>::infinity();
          continue;
        }
        float score = 0.0F;
        for (std::size_t h = 0; h < heads; ++h) {
          const float* q = params.q_idx + (token * heads + h) * dim;
          const float* k = params.k_idx + (b * length + s) * dim;
          float dot = 0.0F;
          for (std::size_t d = 0; d < dim; ++d) {
            dot += q[d] * k[d];
          }
          const float activated = std::max(scale * dot, 0.0F);
          const float weight = params.w_idx == nullptr
                                   ? 1.0F
                                   : params.w_idx[token * heads + h];
          score += weight * activated;
        }
        scores[s] = score;
      }

      TopKParams topk;
      topk.scores = scores.data();
      topk.indices = params.indices + token * top_k;
      topk.counts = row_counts.data();
      topk.q_blocks = 1;
      topk.q_heads = 1;
      topk.kv_blocks = params.sequence_length;
      topk.top_k = params.top_k;
      if (attention::topk(topk) != TopKStatus::kOk) {
        return DsaStatus::kInvalidArgument;
      }
      if (params.counts != nullptr) {
        params.counts[token] = row_counts[0];
      }
    }
  }
  return DsaStatus::kOk;
}

inline DsaStatus dsa(const DsaParams& params) {
  if (params.batch < 0 || params.sequence_length < 0 ||
      params.query_heads <= 0 || params.kv_heads <= 0 ||
      params.query_heads % params.kv_heads != 0 || params.head_dim <= 0 ||
      params.value_dim <= 0 || params.top_k <= 0 ||
      (params.indices == nullptr &&
       (params.index_heads <= 0 || params.index_dim <= 0))) {
    return DsaStatus::kInvalidArgument;
  }
  const std::size_t batch = static_cast<std::size_t>(params.batch);
  const std::size_t length = static_cast<std::size_t>(params.sequence_length);
  const std::size_t query_heads = static_cast<std::size_t>(params.query_heads);
  const std::size_t kv_heads = static_cast<std::size_t>(params.kv_heads);
  const std::size_t head_dim = static_cast<std::size_t>(params.head_dim);
  const std::size_t value_dim = static_cast<std::size_t>(params.value_dim);
  const std::size_t top_k = static_cast<std::size_t>(params.top_k);
  const std::size_t token_count = batch * length;
  if (token_count > 0 &&
      (params.q == nullptr || params.k == nullptr || params.v == nullptr ||
       params.output == nullptr ||
       (params.indices == nullptr &&
        (params.q_idx == nullptr || params.k_idx == nullptr)))) {
    return DsaStatus::kInvalidArgument;
  }

  std::vector<std::int32_t> computed_indices;
  if (params.indices == nullptr && params.selected_indices == nullptr) {
    computed_indices.resize(token_count * top_k);
  }
  std::int32_t* selection = params.indices == nullptr
                                ? (params.selected_indices == nullptr
                                       ? computed_indices.data()
                                       : params.selected_indices)
                                : params.selected_indices;
  if (params.indices == nullptr && token_count > 0) {
    DsaIndexerParams indexer;
    indexer.q_idx = params.q_idx;
    indexer.k_idx = params.k_idx;
    indexer.w_idx = params.w_idx;
    indexer.indices = selection;
    indexer.batch = params.batch;
    indexer.sequence_length = params.sequence_length;
    indexer.index_heads = params.index_heads;
    indexer.index_dim = params.index_dim;
    indexer.top_k = params.top_k;
    indexer.scale = params.index_scale;
    const DsaStatus status = dsa_indexer(indexer);
    if (status != DsaStatus::kOk) {
      return status;
    }
  } else if (params.indices != nullptr && selection != nullptr) {
    std::copy(params.indices, params.indices + token_count * top_k, selection);
  }
  const std::int32_t* indices = params.indices == nullptr ? selection : params.indices;
  const float scale = dsa_default_scale(params.scale, params.head_dim);
  const std::size_t group_size = query_heads / kv_heads;
  std::vector<bool> selected(length);
  std::vector<float> weights(length);

  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t t = 0; t < length; ++t) {
      const std::size_t token = b * length + t;
      std::fill(selected.begin(), selected.end(), false);
      for (std::size_t slot = 0; slot < top_k; ++slot) {
        const std::int32_t s = indices[token * top_k + slot];
        if (s < 0) {
          continue;
        }
        if (static_cast<std::size_t>(s) >= length) {
          return DsaStatus::kIndexOutOfBounds;
        }
        if (static_cast<std::size_t>(s) <= t) {
          selected[static_cast<std::size_t>(s)] = true;
        }
      }

      for (std::size_t h = 0; h < query_heads; ++h) {
        const std::size_t kv_h = h / group_size;
        const float* q = params.q + (token * query_heads + h) * head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t s = 0; s <= t; ++s) {
          if (!selected[s]) {
            continue;
          }
          const float* k = params.k + ((b * length + s) * kv_heads + kv_h) * head_dim;
          float dot = 0.0F;
          for (std::size_t d = 0; d < head_dim; ++d) {
            dot += q[d] * k[d];
          }
          maximum = std::max(maximum, scale * dot);
        }

        float denominator = 0.0F;
        for (std::size_t s = 0; s <= t; ++s) {
          if (!selected[s]) {
            weights[s] = 0.0F;
            continue;
          }
          const float* k = params.k + ((b * length + s) * kv_heads + kv_h) * head_dim;
          float dot = 0.0F;
          for (std::size_t d = 0; d < head_dim; ++d) {
            dot += q[d] * k[d];
          }
          weights[s] = std::exp(scale * dot - maximum);
          denominator += weights[s];
        }

        float* output = params.output +
                        (token * query_heads + h) * value_dim;
        std::fill(output, output + value_dim, 0.0F);
        if (denominator == 0.0F) {
          continue;
        }
        for (std::size_t s = 0; s <= t; ++s) {
          if (!selected[s]) {
            continue;
          }
          const float probability = weights[s] / denominator;
          const float* v = params.v +
                           ((b * length + s) * kv_heads + kv_h) * value_dim;
          for (std::size_t d = 0; d < value_dim; ++d) {
            output[d] += probability * v[d];
          }
        }
      }
    }
  }
  return DsaStatus::kOk;
}

inline DsaStatus naive_dsa_indexer(const DsaIndexerParams& params) {
  return dsa_indexer(params);
}

inline DsaStatus naive_dsa(const DsaParams& params) {
  return dsa(params);
}

}  // namespace attention

#endif  // ATTENTION_DSA_HPP_
