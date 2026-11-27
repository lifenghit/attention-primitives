#include "attention/msa.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

int failures = 0;

#define EXPECT_TRUE(condition)                                                \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::cerr << __FILE__ << ':' << __LINE__ << ": expected " #condition   \
                << '\n';                                                      \
      ++failures;                                                             \
    }                                                                         \
  } while (false)

#define EXPECT_EQ(actual, expected) EXPECT_TRUE((actual) == (expected))

bool near(float actual, float expected) {
  return std::fabs(actual - expected) <= 1e-5F;
}

float reference_output(std::size_t query, std::size_t group,
                       const std::vector<std::int32_t>& indices,
                       const std::vector<std::int32_t>& counts,
                       std::size_t top_k, std::size_t kv_heads) {
  const std::size_t offset = (query * kv_heads + group) * top_k;
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t slot = 0;
       slot < static_cast<std::size_t>(counts[query * kv_heads + group]); ++slot) {
    const std::size_t block = static_cast<std::size_t>(indices[offset + slot]);
    const std::size_t begin = block * 2;
    const std::size_t end = std::min(begin + 2, query + 1);
    for (std::size_t key = begin; key < end; ++key) {
      maximum = std::max(maximum, static_cast<float>(key + 1));
    }
  }
  float numerator = 0.0F;
  float denominator = 0.0F;
  for (std::size_t slot = 0;
       slot < static_cast<std::size_t>(counts[query * kv_heads + group]); ++slot) {
    const std::size_t block = static_cast<std::size_t>(indices[offset + slot]);
    const std::size_t begin = block * 2;
    const std::size_t end = std::min(begin + 2, query + 1);
    for (std::size_t key = begin; key < end; ++key) {
      const float weight = std::exp(static_cast<float>(key + 1) - maximum);
      denominator += weight;
      numerator += weight * (10.0F * static_cast<float>(key) +
                             100.0F * static_cast<float>(group));
    }
  }
  return numerator / denominator;
}

struct Fixture {
  static constexpr std::size_t t = 5;
  static constexpr std::size_t h = 4;
  static constexpr std::size_t hv = 2;

  std::vector<float> q = std::vector<float>(t * h, 1.0F);
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> q_idx = std::vector<float>(t * h);
  std::vector<float> k_idx = {1.0F, 3.0F, 2.0F, 4.0F, -2.0F};

  Fixture() {
    for (std::size_t key = 0; key < t; ++key) {
      for (std::size_t group = 0; group < hv; ++group) {
        k.push_back(static_cast<float>(key + 1));
        v.push_back(10.0F * static_cast<float>(key) +
                    100.0F * static_cast<float>(group));
      }
      for (std::size_t head = 0; head < h; ++head) {
        q_idx[key * h + head] = head < 2 ? 1.0F : -1.0F;
      }
    }
  }

  attention::MSAParams params(std::vector<float>& output,
                              std::vector<std::int32_t>& indices,
                              std::vector<std::int32_t>& counts,
                              std::int32_t top_k) const {
    attention::MSAParams result;
    result.q = q.data();
    result.k = k.data();
    result.v = v.data();
    result.q_idx = q_idx.data();
    result.k_idx = k_idx.data();
    result.output = output.data();
    result.indices = indices.data();
    result.counts = counts.data();
    result.sequence_length = static_cast<std::int32_t>(t);
    result.q_heads = static_cast<std::int32_t>(h);
    result.kv_heads = static_cast<std::int32_t>(hv);
    result.d = 1;
    result.v_dim = 1;
    result.index_dim = 1;
    result.block_size = 2;
    result.topk_blocks = top_k;
    return result;
  }
};

void test_group_independent_selection_forced_current_and_attention() {
  constexpr std::size_t top_k = 2;
  Fixture fixture;
  std::vector<float> output(Fixture::t * Fixture::h, -1.0F);
  std::vector<std::int32_t> indices(Fixture::t * Fixture::hv * top_k, -99);
  std::vector<std::int32_t> counts(Fixture::t * Fixture::hv, -99);
  auto params = fixture.params(output, indices, counts, top_k);
  EXPECT_EQ(attention::msa(params), attention::MSAStatus::kOk);

  const std::size_t last = 4 * Fixture::hv * top_k;
  EXPECT_EQ(indices[last], 1);
  EXPECT_EQ(indices[last + 1], 2);
  EXPECT_EQ(indices[last + 2], 2);
  EXPECT_EQ(indices[last + 3], 0);
  EXPECT_EQ(counts[8], 2);
  EXPECT_EQ(counts[9], 2);

  for (std::size_t query = 0; query < Fixture::t; ++query) {
    for (std::size_t group = 0; group < Fixture::hv; ++group) {
      const float expected = reference_output(query, group, indices, counts,
                                               top_k, Fixture::hv);
      for (std::size_t local_head = 0; local_head < 2; ++local_head) {
        const std::size_t head = group * 2 + local_head;
        EXPECT_TRUE(near(output[query * Fixture::h + head], expected));
      }
    }
  }
}

void test_non_full_block_topk_padding_and_stable_ties() {
  constexpr std::size_t top_k = 5;
  Fixture fixture;
  std::vector<float> output(Fixture::t * Fixture::h);
  std::vector<std::int32_t> indices(Fixture::t * Fixture::hv * top_k, -99);
  std::vector<std::int32_t> counts(Fixture::t * Fixture::hv, -99);
  auto params = fixture.params(output, indices, counts, top_k);
  EXPECT_EQ(attention::msa(params), attention::MSAStatus::kOk);

  const std::size_t last = 4 * Fixture::hv * top_k;
  const std::int32_t expected[] = {1, 0, 2, -1, -1, 2, 0, 1, -1, -1};
  for (std::size_t slot = 0; slot < 10; ++slot) {
    EXPECT_EQ(indices[last + slot], expected[slot]);
  }
  EXPECT_EQ(counts[8], 3);
  EXPECT_EQ(counts[9], 3);

  std::fill(fixture.q_idx.begin(), fixture.q_idx.end(), 0.0F);
  params = fixture.params(output, indices, counts, 2);
  EXPECT_EQ(attention::msa(params), attention::MSAStatus::kOk);
  const std::size_t tied_last = 4 * Fixture::hv * 2;
  EXPECT_EQ(indices[tied_last], 0);
  EXPECT_EQ(indices[tied_last + 1], 2);
  EXPECT_EQ(indices[tied_last + 2], 0);
  EXPECT_EQ(indices[tied_last + 3], 2);
}

void test_invalid_arguments() {
  Fixture fixture;
  std::vector<float> output(Fixture::t * Fixture::h);
  std::vector<std::int32_t> indices(Fixture::t * Fixture::hv * 2);
  std::vector<std::int32_t> counts(Fixture::t * Fixture::hv);
  auto params = fixture.params(output, indices, counts, 2);
  params.q_heads = 3;
  EXPECT_EQ(attention::msa(params), attention::MSAStatus::kInvalidArgument);
  params = fixture.params(output, indices, counts, 2);
  params.topk_blocks = 0;
  EXPECT_EQ(attention::msa(params), attention::MSAStatus::kInvalidArgument);
  params = fixture.params(output, indices, counts, 2);
  params.output = nullptr;
  EXPECT_EQ(attention::msa(params), attention::MSAStatus::kInvalidArgument);
}

}  // namespace

int main() {
  test_group_independent_selection_forced_current_and_attention();
  test_non_full_block_topk_padding_and_stable_ties();
  test_invalid_arguments();
  if (failures != 0) {
    std::cerr << failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << "All MSA CPU tests passed\n";
  return 0;
}
