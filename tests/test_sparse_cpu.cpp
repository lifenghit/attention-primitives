#include "attention/block_score.hpp"
#include "attention/block_selection.hpp"
#include "attention/topk.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
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

attention::Float16 fp16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  std::uint32_t exponent = (bits >> 23) & 0xffu;
  std::uint32_t fraction = bits & 0x7fffffu;
  if (exponent == 0xffu) {
    return {static_cast<std::uint16_t>(sign | 0x7c00u | (fraction >> 13))};
  }
  int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 0x1f) {
    return {static_cast<std::uint16_t>(sign | 0x7c00u)};
  }
  if (half_exponent <= 0) {
    return {static_cast<std::uint16_t>(sign)};
  }
  return {static_cast<std::uint16_t>(sign |
                                    (static_cast<std::uint32_t>(half_exponent)
                                     << 10) |
                                    (fraction >> 13))};
}

std::vector<attention::Float16> fp16_vector(
    const std::vector<float>& values) {
  std::vector<attention::Float16> result;
  result.reserve(values.size());
  for (float value : values) {
    result.push_back(fp16(value));
  }
  return result;
}

bool near(float actual, float expected) {
  return std::fabs(actual - expected) <= 1e-6F;
}

void test_fp16_conversion() {
  EXPECT_EQ(fp16_to_float(fp16(0.0F)), 0.0F);
  EXPECT_EQ(fp16_to_float(fp16(-2.0F)), -2.0F);
  EXPECT_EQ(fp16_to_float(fp16(0.5F)), 0.5F);
  EXPECT_TRUE(std::isinf(attention::fp16_to_float({0x7c00u})));
  EXPECT_TRUE(std::isnan(attention::fp16_to_float({0x7e00u})));
  EXPECT_TRUE(near(attention::fp16_to_float({0x0001u}), std::ldexp(1.0F, -24)));
}

void test_block_score_gqa_and_causal() {
  const auto q = fp16_vector({
      2, 4, 0, 2, 4, 4, 1, 3,
      0, 0, 0, 0, 0, 0, 0, 0,
  });
  const auto k = fp16_vector({
      2, 2, 4, 0,
      1, 3, 2, 6,
      8, 8, 8, 8,
  });
  std::vector<float> scores(2 * 4 * 3, 0.0F);
  attention::BlockScoreParams params;
  params.q = q.data();
  params.k = k.data();
  params.scores = scores.data();
  params.q_blocks = 2;
  params.q_heads = 4;
  params.kv_blocks = 3;
  params.kv_heads = 2;
  params.d = 2;
  params.causal = true;
  EXPECT_EQ(attention::block_score(params), attention::BlockScoreStatus::kOk);

  const float scale = 1.0F / std::sqrt(2.0F);
  EXPECT_TRUE(near(scores[0], 6.0F * scale));
  EXPECT_EQ(scores[1], -INFINITY);
  EXPECT_EQ(scores[2], -INFINITY);
  EXPECT_TRUE(near(scores[3], 2.0F * scale));
  EXPECT_TRUE(near(scores[6], 8.0F * scale));
  EXPECT_TRUE(near(scores[9], 4.0F * scale));
  EXPECT_TRUE(near(scores[12], 0.0F));
  EXPECT_TRUE(near(scores[15], 0.0F));
  EXPECT_EQ(scores[4], -INFINITY);
  EXPECT_EQ(scores[7], -INFINITY);
  EXPECT_EQ(scores[10], -INFINITY);
  EXPECT_EQ(scores[23], -INFINITY);

  auto invalid = params;
  invalid.q_heads = 3;
  EXPECT_EQ(attention::block_score(invalid),
            attention::BlockScoreStatus::kInvalidArgument);
  invalid = params;
  invalid.d = 0;
  EXPECT_EQ(attention::block_score(invalid),
            attention::BlockScoreStatus::kInvalidArgument);
}

void test_topk_ties_limit_and_empty_row() {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  const std::vector<float> scores = {
      1.0F, 3.0F, 3.0F,
      negative_infinity, negative_infinity, negative_infinity,
  };
  std::vector<std::int32_t> indices(2 * 5, 99);
  std::vector<std::int32_t> counts(2, 99);
  attention::TopKParams params;
  params.scores = scores.data();
  params.indices = indices.data();
  params.counts = counts.data();
  params.q_blocks = 1;
  params.q_heads = 2;
  params.kv_blocks = 3;
  params.top_k = 5;
  EXPECT_EQ(attention::topk(params), attention::TopKStatus::kOk);
  EXPECT_EQ(counts[0], 3);
  EXPECT_EQ(counts[1], 0);
  EXPECT_EQ(indices[0], 1);
  EXPECT_EQ(indices[1], 2);
  EXPECT_EQ(indices[2], 0);
  for (std::size_t i = 3; i < indices.size(); ++i) {
    EXPECT_EQ(indices[i], -1);
  }

  const std::vector<float> one_row = {2.0F, negative_infinity, 2.0F};
  params.scores = one_row.data();
  params.q_heads = 1;
  params.top_k = 2;
  EXPECT_EQ(attention::topk(params), attention::TopKStatus::kOk);
  EXPECT_EQ(counts[0], 2);
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(indices[1], 2);

  params.top_k = 0;
  EXPECT_EQ(attention::topk(params), attention::TopKStatus::kInvalidArgument);
}

void test_block_selection_mapping() {
  const std::vector<std::int32_t> indices = {
      1, 2, 0, -1, -1,
      -1, -1, -1, -1, -1,
  };
  const std::vector<std::int32_t> counts = {3, 0};
  const std::vector<std::int32_t> table = {7, 4, 9};
  std::vector<std::int32_t> physical(10, 99);
  std::vector<std::int32_t> positions(10, 99);
  std::vector<std::int32_t> output_counts(2, 99);
  attention::BlockSelectionParams params;
  params.topk_indices = indices.data();
  params.counts = counts.data();
  params.block_table = table.data();
  params.physical_blocks = physical.data();
  params.dense_positions = positions.data();
  params.output_counts = output_counts.data();
  params.q_blocks = 1;
  params.q_heads = 2;
  params.kv_blocks = 3;
  params.top_k = 5;
  params.physical_block_count = 10;
  EXPECT_EQ(attention::block_selection(params),
            attention::BlockSelectionStatus::kOk);
  EXPECT_EQ(output_counts[0], 3);
  EXPECT_EQ(output_counts[1], 0);
  EXPECT_EQ(physical[0], 4);
  EXPECT_EQ(physical[1], 9);
  EXPECT_EQ(physical[2], 7);
  EXPECT_EQ(positions[0], 0);
  EXPECT_EQ(positions[1], 1);
  EXPECT_EQ(positions[2], 2);
  for (std::size_t i = 3; i < physical.size(); ++i) {
    EXPECT_EQ(physical[i], -1);
    EXPECT_EQ(positions[i], -1);
  }

  params.block_table = nullptr;
  EXPECT_EQ(attention::block_selection(params),
            attention::BlockSelectionStatus::kOk);
  EXPECT_EQ(physical[0], 1);
  EXPECT_EQ(physical[1], 2);
  EXPECT_EQ(physical[2], 0);

  const std::vector<std::int32_t> bad_table = {7, 10, 9};
  params.block_table = bad_table.data();
  physical.assign(10, 99);
  EXPECT_EQ(attention::block_selection(params),
            attention::BlockSelectionStatus::kIndexOutOfBounds);
  EXPECT_EQ(physical[0], 99);

  const std::vector<std::int32_t> bad_indices = {3, 2, 0, -1, -1,
                                                 -1, -1, -1, -1, -1};
  params.topk_indices = bad_indices.data();
  EXPECT_EQ(attention::block_selection(params),
            attention::BlockSelectionStatus::kIndexOutOfBounds);
}

}  // namespace

int main() {
  test_fp16_conversion();
  test_block_score_gqa_and_causal();
  test_topk_ties_limit_and_empty_row();
  test_block_selection_mapping();
  if (failures != 0) {
    std::cerr << failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << "All sparse CPU tests passed\n";
  return 0;
}
