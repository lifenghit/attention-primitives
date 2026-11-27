#ifndef ATTENTION_MSA_HPP_
#define ATTENTION_MSA_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace attention {

enum class MSAStatus {
  kOk,
  kInvalidArgument,
};

struct MSAParams {
  const float* q = nullptr;
  const float* k = nullptr;
  const float* v = nullptr;
  const float* q_idx = nullptr;
  const float* k_idx = nullptr;
  float* output = nullptr;
  std::int32_t* indices = nullptr;
  std::int32_t* counts = nullptr;
  std::int32_t sequence_length = 0;
  std::int32_t q_heads = 0;
  std::int32_t kv_heads = 0;
  std::int32_t d = 0;
  std::int32_t v_dim = 0;
  std::int32_t index_dim = 0;
  std::int32_t block_size = 128;
  std::int32_t topk_blocks = 16;
  float query_scale = 0.0F;
  float index_scale = 0.0F;
};

inline MSAStatus msa(const MSAParams& params) {
  if (params.sequence_length < 0 || params.q_heads <= 0 ||
      params.kv_heads <= 0 || params.q_heads % params.kv_heads != 0 ||
      params.d <= 0 || params.v_dim <= 0 || params.index_dim <= 0 ||
      params.block_size <= 0 || params.topk_blocks <= 0) {
    return MSAStatus::kInvalidArgument;
  }
  if (params.sequence_length > 0 &&
      (params.q == nullptr || params.k == nullptr || params.v == nullptr ||
       params.q_idx == nullptr || params.k_idx == nullptr ||
       params.output == nullptr || params.indices == nullptr ||
       params.counts == nullptr)) {
    return MSAStatus::kInvalidArgument;
  }

  const std::size_t t = static_cast<std::size_t>(params.sequence_length);
  const std::size_t h = static_cast<std::size_t>(params.q_heads);
  const std::size_t hv = static_cast<std::size_t>(params.kv_heads);
  const std::size_t d = static_cast<std::size_t>(params.d);
  const std::size_t dv = static_cast<std::size_t>(params.v_dim);
  const std::size_t di = static_cast<std::size_t>(params.index_dim);
  const std::size_t block_size = static_cast<std::size_t>(params.block_size);
  const std::size_t top_k = static_cast<std::size_t>(params.topk_blocks);
  const std::size_t group_size = h / hv;
  const float query_scale = params.query_scale == 0.0F
                                ? 1.0F / std::sqrt(static_cast<float>(params.d))
                                : params.query_scale;
  const float index_scale = params.index_scale == 0.0F
                                ? 1.0F / std::sqrt(static_cast<float>(params.index_dim))
                                : params.index_scale;

  std::vector<float> token_scores;
  std::vector<float> block_scores;
  std::vector<std::int32_t> order;
  std::vector<bool> selected_blocks;
  for (std::size_t query = 0; query < t; ++query) {
    const std::size_t current_block = query / block_size;
    const std::size_t visible_blocks = current_block + 1;
    token_scores.resize(query + 1);
    block_scores.resize(visible_blocks);
    order.resize(visible_blocks);

    for (std::size_t group = 0; group < hv; ++group) {
      selected_blocks.assign(visible_blocks, false);
      for (std::size_t key = 0; key <= query; ++key) {
        float score = 0.0F;
        for (std::size_t local_head = 0; local_head < group_size; ++local_head) {
          const std::size_t head = group * group_size + local_head;
          const float* q_index = params.q_idx + (query * h + head) * di;
          const float* k_index = params.k_idx + key * di;
          float dot = 0.0F;
          for (std::size_t channel = 0; channel < di; ++channel) {
            dot += q_index[channel] * k_index[channel];
          }
          score += dot * index_scale;
        }
        token_scores[key] = score;
      }

      for (std::size_t block = 0; block < visible_blocks; ++block) {
        const std::size_t begin = block * block_size;
        const std::size_t end = std::min(begin + block_size, query + 1);
        block_scores[block] = token_scores[begin];
        for (std::size_t key = begin + 1; key < end; ++key) {
          block_scores[block] = std::max(block_scores[block], token_scores[key]);
        }
      }

      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](std::int32_t lhs,
                                                std::int32_t rhs) {
        const float lhs_score = block_scores[static_cast<std::size_t>(lhs)];
        const float rhs_score = block_scores[static_cast<std::size_t>(rhs)];
        return lhs_score > rhs_score ||
               (lhs_score == rhs_score && lhs < rhs);
      });
      std::size_t count = std::min(top_k, visible_blocks);
      std::int32_t* indices = params.indices + (query * hv + group) * top_k;
      for (std::size_t slot = 0; slot < count; ++slot) {
        indices[slot] = order[slot];
        selected_blocks[static_cast<std::size_t>(order[slot])] = true;
      }
      if (!selected_blocks[current_block]) {
        if (count < top_k) {
          indices[count] = static_cast<std::int32_t>(current_block);
          ++count;
        } else {
          selected_blocks[static_cast<std::size_t>(indices[count - 1])] = false;
          indices[count - 1] = static_cast<std::int32_t>(current_block);
        }
        selected_blocks[current_block] = true;
      }
      params.counts[query * hv + group] = static_cast<std::int32_t>(count);
      for (std::size_t slot = count; slot < top_k; ++slot) {
        indices[slot] = -1;
      }

      for (std::size_t local_head = 0; local_head < group_size; ++local_head) {
        const std::size_t head = group * group_size + local_head;
        const float* q = params.q + (query * h + head) * d;
        float maximum = -INFINITY;
        for (std::size_t key = 0; key <= query; ++key) {
          if (!selected_blocks[key / block_size]) {
            continue;
          }
          const float* k = params.k + (key * hv + group) * d;
          float score = 0.0F;
          for (std::size_t channel = 0; channel < d; ++channel) {
            score += q[channel] * k[channel];
          }
          maximum = std::max(maximum, score * query_scale);
        }

        float denominator = 0.0F;
        float* output = params.output + (query * h + head) * dv;
        for (std::size_t channel = 0; channel < dv; ++channel) {
          output[channel] = 0.0F;
        }
        for (std::size_t key = 0; key <= query; ++key) {
          if (!selected_blocks[key / block_size]) {
            continue;
          }
          const float* k = params.k + (key * hv + group) * d;
          float score = 0.0F;
          for (std::size_t channel = 0; channel < d; ++channel) {
            score += q[channel] * k[channel];
          }
          const float weight = std::exp(score * query_scale - maximum);
          denominator += weight;
          const float* v = params.v + (key * hv + group) * dv;
          for (std::size_t channel = 0; channel < dv; ++channel) {
            output[channel] += weight * v[channel];
          }
        }
        for (std::size_t channel = 0; channel < dv; ++channel) {
          output[channel] /= denominator;
        }
      }
    }
  }
  return MSAStatus::kOk;
}

}  // namespace attention

#endif  // ATTENTION_MSA_HPP_
