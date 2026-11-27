#include "attention/qsa.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
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

bool near(float actual, float expected, float tolerance = 1e-5F) {
  return std::fabs(actual - expected) <= tolerance;
}

void test_visible_ratio_boundaries() {
  const std::vector<std::int32_t> expected = {0, 1, 1, 2, 2};
  for (std::int32_t position = 0; position < 5; ++position) {
    std::int32_t visible = -1;
    EXPECT_EQ(attention::qsa_visible_blocks(position, 5, 2, &visible),
              attention::QSAStatus::kOk);
    EXPECT_EQ(visible, expected[position]);
  }
  std::int32_t visible = -1;
  EXPECT_EQ(attention::qsa_visible_blocks(0, 3, 4, &visible),
            attention::QSAStatus::kOk);
  EXPECT_EQ(visible, 0);
  EXPECT_EQ(attention::qsa_visible_blocks(2, 3, 1, &visible),
            attention::QSAStatus::kOk);
  EXPECT_EQ(visible, 3);
  EXPECT_EQ(attention::qsa_visible_blocks(0, 3, 0, &visible),
            attention::QSAStatus::kInvalidArgument);
  EXPECT_EQ(attention::qsa_visible_blocks(3, 3, 2, &visible),
            attention::QSAStatus::kInvalidArgument);
}

void test_group_compression_discards_partial_tail() {
  const std::vector<float> input = {
      1, 10, 3, 30, 5, 50,
      2, 20, 4, 40, 6, 60,
      100, 1000, 200, 2000, 300, 3000,
  };
  std::vector<float> output(1 * 2 * 3, -1.0F);
  attention::QSACompressParams params;
  params.input = input.data();
  params.output = output.data();
  params.sequence_length = 3;
  params.heads = 2;
  params.head_dim = 3;
  params.compress_ratio = 2;
  EXPECT_EQ(attention::qsa_compress_groups(params), attention::QSAStatus::kOk);
  const std::vector<float> expected = {1.5F, 15, 3.5F, 35, 5.5F, 55};
  for (std::size_t i = 0; i < output.size(); ++i) {
    EXPECT_TRUE(near(output[i], expected[i]));
  }
  params.compress_ratio = 0;
  EXPECT_EQ(attention::qsa_compress_groups(params),
            attention::QSAStatus::kInvalidArgument);
}

attention::QSAScoreParams score_fixture(
    const std::vector<float>& index_q, const std::vector<float>& keys,
    const std::vector<std::int32_t>& table, std::vector<float>& scores,
    std::vector<std::int32_t>& visible) {
  attention::QSAScoreParams params;
  params.index_q = index_q.data();
  params.compressed_k = keys.data();
  params.block_table = table.data();
  params.scores = scores.data();
  params.visible_blocks = visible.data();
  params.sequence_length = 5;
  params.index_heads = 2;
  params.head_dim = 1;
  params.logical_blocks = 2;
  params.physical_block_count = 2;
  params.compress_ratio = 2;
  params.score_divisor = 1.0F;
  return params;
}

void test_multi_head_relu_scores_and_stable_selection() {
  const std::vector<float> index_q = {
      0, 0, 0, 0, 2, -1, 1, 1, 1, 1,
  };
  const std::vector<float> keys = {2, 3};
  const std::vector<std::int32_t> table = {1, 0};
  std::vector<float> scores(5 * 2, 99);
  std::vector<std::int32_t> visible(5, -1);
  auto params = score_fixture(index_q, keys, table, scores, visible);
  EXPECT_EQ(attention::qsa_scores(params), attention::QSAStatus::kOk);
  EXPECT_EQ(scores[0], -INFINITY);
  EXPECT_EQ(scores[4], 6.0F);
  EXPECT_EQ(scores[5], -INFINITY);
  EXPECT_EQ(scores[6], 6.0F);
  EXPECT_EQ(scores[7], 4.0F);
  EXPECT_EQ(scores[8], 6.0F);
  EXPECT_EQ(scores[9], 4.0F);

  const std::vector<float> tied_q = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const std::vector<float> tied_keys = {2, 2};
  params = score_fixture(tied_q, tied_keys, table, scores, visible);
  EXPECT_EQ(attention::qsa_scores(params), attention::QSAStatus::kOk);
  EXPECT_EQ(scores[6], 4.0F);
  EXPECT_EQ(scores[7], 4.0F);
}

void test_physical_page_errors_only_when_visible() {
  const std::vector<float> index_q(10, 1.0F);
  const std::vector<float> keys = {2, 3};
  const std::vector<std::int32_t> bad_negative = {-1, 0};
  std::vector<float> scores(10, 0);
  std::vector<std::int32_t> visible(5, 0);
  auto params = score_fixture(index_q, keys, bad_negative, scores, visible);
  EXPECT_EQ(attention::qsa_scores(params),
            attention::QSAStatus::kPhysicalPageOutOfBounds);

  const std::vector<std::int32_t> bad_large = {2, 0};
  params = score_fixture(index_q, keys, bad_large, scores, visible);
  EXPECT_EQ(attention::qsa_scores(params),
            attention::QSAStatus::kPhysicalPageOutOfBounds);

  const std::vector<std::int32_t> unused_bad_page = {0, -1};
  params = score_fixture(index_q, keys, unused_bad_page, scores, visible);
  params.sequence_length = 3;
  params.logical_blocks = 1;
  params.physical_block_count = 1;
  EXPECT_EQ(attention::qsa_scores(params), attention::QSAStatus::kOk);
}

void test_end_to_end_paged_qsa_and_fp32_softmax() {
  const std::vector<float> q(5, 1.0F);
  const std::vector<float> k = {1, 3, 2, 4, 100};
  const std::vector<float> v = {10, 30, 30, 50, 1000};
  const std::vector<float> index_q = {
      0, 0, 0, 0, 2, -1, 1, 1, 1, 1,
  };
  const std::vector<float> index_k = {1, 3, 2, 4, 100};
  const std::vector<std::int32_t> table = {1, 0};
  std::vector<float> output(5, -1);
  std::vector<float> scores(10, -1);
  std::vector<std::int32_t> visible(5, -1);
  std::vector<std::int32_t> selected(10, -9);
  std::vector<std::int32_t> physical(10, -9);
  std::vector<std::int32_t> counts(5, -1);

  attention::QSAParams params;
  params.q = q.data();
  params.k = k.data();
  params.v = v.data();
  params.index_q = index_q.data();
  params.index_k = index_k.data();
  params.block_table = table.data();
  params.output = output.data();
  params.scores = scores.data();
  params.visible_blocks = visible.data();
  params.selected_indices = selected.data();
  params.physical_blocks = physical.data();
  params.selected_counts = counts.data();
  params.sequence_length = 5;
  params.heads = 1;
  params.kv_heads = 1;
  params.index_heads = 2;
  params.head_dim = 1;
  params.value_dim = 1;
  params.index_dim = 1;
  params.physical_block_count = 2;
  params.compress_ratio = 2;
  params.top_k = 2;
  params.score_divisor = 1.0F;
  params.query_scale = 1.0F;
  EXPECT_EQ(attention::qsa(params), attention::QSAStatus::kOk);
  EXPECT_EQ(output[0], 0.0F);
  EXPECT_EQ(output[1], 40.0F);
  EXPECT_EQ(output[2], 40.0F);
  const float expected = (40.0F * std::exp(3.0F) + 20.0F * std::exp(2.0F)) /
                         (std::exp(3.0F) + std::exp(2.0F));
  EXPECT_TRUE(near(output[3], expected));
  EXPECT_TRUE(near(output[4], expected));
  EXPECT_EQ(counts[0], 0);
  EXPECT_EQ(counts[2], 1);
  EXPECT_EQ(counts[3], 2);
  EXPECT_EQ(selected[2 * 2], 0);
  EXPECT_EQ(selected[2 * 2 + 1], -1);
  EXPECT_EQ(physical[2 * 2], 1);
  EXPECT_EQ(physical[2 * 2 + 1], -1);
  EXPECT_EQ(selected[3 * 2], 0);
  EXPECT_EQ(selected[3 * 2 + 1], 1);
  EXPECT_EQ(physical[3 * 2], 1);
  EXPECT_EQ(physical[3 * 2 + 1], 0);

  const std::vector<float> tied_index_q(10, 1.0F);
  const std::vector<float> tied_index_k(5, 2.0F);
  params.index_q = tied_index_q.data();
  params.index_k = tied_index_k.data();
  EXPECT_EQ(attention::qsa(params), attention::QSAStatus::kOk);
  EXPECT_EQ(selected[3 * 2], 0);
  EXPECT_EQ(selected[3 * 2 + 1], 1);
  EXPECT_EQ(physical[3 * 2], 1);
  EXPECT_EQ(physical[3 * 2 + 1], 0);

  const std::vector<std::int32_t> bad_table = {-1, 0};
  params.block_table = bad_table.data();
  EXPECT_EQ(attention::qsa(params),
            attention::QSAStatus::kPhysicalPageOutOfBounds);
  params.block_table = table.data();
  params.compress_ratio = 0;
  EXPECT_EQ(attention::qsa(params), attention::QSAStatus::kInvalidArgument);
}

}  // namespace

int main() {
  test_visible_ratio_boundaries();
  test_group_compression_discards_partial_tail();
  test_multi_head_relu_scores_and_stable_selection();
  test_physical_page_errors_only_when_visible();
  test_end_to_end_paged_qsa_and_fp32_softmax();
  if (failures != 0) {
    std::cerr << failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << "All QSA CPU tests passed\n";
  return 0;
}
