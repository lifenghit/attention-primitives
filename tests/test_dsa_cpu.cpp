#include "attention/dsa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void expectTrue(bool condition) {
  if (!condition) {
    std::cerr << "DSA CPU assertion failed\n";
    std::exit(1);
  }
}

void expectNear(float actual, float expected, float tolerance = 1.0e-5F) {
  const float scale = std::max({1.0F, std::fabs(actual), std::fabs(expected)});
  if (std::fabs(actual - expected) > tolerance * scale) {
    std::cerr << "expected " << expected << ", got " << actual << '\n';
    std::exit(1);
  }
}

void testWeightedReluStableTopKAndPadding() {
  const std::vector<float> q_idx = {
      1.0F, 2.0F,
      2.0F, -1.0F,
      1.0F, 1.0F,
  };
  const std::vector<float> k_idx = {1.0F, -2.0F, 3.0F};
  const std::vector<float> w_idx = {
      0.5F, 2.0F,
      1.0F, 1.0F,
      1.0F, 1.0F,
  };
  std::vector<std::int32_t> indices(3 * 5, 99);
  std::vector<std::int32_t> counts(3, 99);
  attention::DsaIndexerParams params;
  params.q_idx = q_idx.data();
  params.k_idx = k_idx.data();
  params.w_idx = w_idx.data();
  params.indices = indices.data();
  params.counts = counts.data();
  params.batch = 1;
  params.sequence_length = 3;
  params.index_heads = 2;
  params.index_dim = 1;
  params.top_k = 5;
  params.scale = 1.0F;
  expectTrue(attention::dsa_indexer(params) == attention::DsaStatus::kOk);

  const std::vector<std::int32_t> expected = {
      0, -1, -1, -1, -1,
      0, 1, -1, -1, -1,
      2, 0, 1, -1, -1,
  };
  expectTrue(indices == expected);
  expectTrue(counts == std::vector<std::int32_t>({1, 2, 3}));
}

void testSharedSelectionAndStableGqaSoftmax() {
  const std::vector<float> q = {
      1000.0F, 1000.0F,
      1000.0F, 1000.0F,
  };
  const std::vector<float> k = {1000.0F, 1000.0F};
  const std::vector<float> v = {
      2.0F, 4.0F,
      10.0F, 20.0F,
  };
  const std::vector<std::int32_t> indices = {
      0, -1, -1,
      0, 1, -1,
  };
  std::vector<float> output(2 * 2 * 2, -1.0F);
  attention::DsaParams params;
  params.q = q.data();
  params.k = k.data();
  params.v = v.data();
  params.indices = indices.data();
  params.output = output.data();
  params.batch = 1;
  params.sequence_length = 2;
  params.query_heads = 2;
  params.kv_heads = 1;
  params.head_dim = 1;
  params.value_dim = 2;
  params.top_k = 3;
  params.scale = 1.0e-6F;
  expectTrue(attention::dsa(params) == attention::DsaStatus::kOk);
  expectNear(output[0], 2.0F);
  expectNear(output[1], 4.0F);
  expectNear(output[2], 2.0F);
  expectNear(output[3], 4.0F);
  for (std::size_t h = 0; h < 2; ++h) {
    expectNear(output[4 + h * 2], 6.0F);
    expectNear(output[5 + h * 2], 12.0F);
  }
}

void testGqaHeadMapping() {
  const std::vector<float> q = {1.0F, 1.0F, 1.0F, 1.0F};
  const std::vector<float> k = {1.0F, 2.0F};
  const std::vector<float> v = {2.0F, 8.0F};
  const std::vector<std::int32_t> indices = {0, 0};
  std::vector<float> output(4, 0.0F);
  attention::DsaParams params;
  params.q = q.data();
  params.k = k.data();
  params.v = v.data();
  params.indices = indices.data();
  params.output = output.data();
  params.batch = 1;
  params.sequence_length = 1;
  params.query_heads = 4;
  params.kv_heads = 2;
  params.head_dim = 1;
  params.value_dim = 1;
  params.top_k = 2;
  params.scale = 1.0F;
  expectTrue(attention::dsa(params) == attention::DsaStatus::kOk);
  expectTrue(output == std::vector<float>({2.0F, 2.0F, 8.0F, 8.0F}));
}

void testEmptySelectionAndIntegratedIndexer() {
  const std::vector<float> q = {1.0F};
  const std::vector<float> k = {1.0F};
  const std::vector<float> v = {7.0F};
  const std::vector<std::int32_t> empty = {-1, -1, -1};
  std::vector<float> output(1, 9.0F);
  attention::DsaParams params;
  params.q = q.data();
  params.k = k.data();
  params.v = v.data();
  params.indices = empty.data();
  params.output = output.data();
  params.batch = 1;
  params.sequence_length = 1;
  params.query_heads = 1;
  params.kv_heads = 1;
  params.head_dim = 1;
  params.value_dim = 1;
  params.top_k = 3;
  expectTrue(attention::dsa(params) == attention::DsaStatus::kOk);
  expectNear(output[0], 0.0F);

  const std::vector<float> q_idx = {2.0F};
  const std::vector<float> k_idx = {3.0F};
  std::vector<std::int32_t> selected(4, 99);
  params.indices = nullptr;
  params.q_idx = q_idx.data();
  params.k_idx = k_idx.data();
  params.index_heads = 1;
  params.index_dim = 1;
  params.top_k = 4;
  params.selected_indices = selected.data();
  expectTrue(attention::dsa(params) == attention::DsaStatus::kOk);
  expectTrue(selected == std::vector<std::int32_t>({0, -1, -1, -1}));
  expectNear(output[0], 7.0F);
}

void testInvalidArgumentsAndIndices() {
  attention::DsaIndexerParams indexer;
  expectTrue(attention::dsa_indexer(indexer) == attention::DsaStatus::kInvalidArgument);

  const std::vector<float> data = {1.0F};
  const std::vector<std::int32_t> bad = {1};
  std::vector<float> output(1, 0.0F);
  attention::DsaParams params;
  params.q = data.data();
  params.k = data.data();
  params.v = data.data();
  params.indices = bad.data();
  params.output = output.data();
  params.batch = 1;
  params.sequence_length = 1;
  params.query_heads = 1;
  params.kv_heads = 1;
  params.head_dim = 1;
  params.value_dim = 1;
  params.top_k = 1;
  expectTrue(attention::dsa(params) == attention::DsaStatus::kIndexOutOfBounds);
  params.kv_heads = 2;
  expectTrue(attention::dsa(params) == attention::DsaStatus::kInvalidArgument);
}

}  // namespace

int main() {
  testWeightedReluStableTopKAndPadding();
  testSharedSelectionAndStableGqaSoftmax();
  testGqaHeadMapping();
  testEmptySelectionAndIntegratedIndexer();
  testInvalidArgumentsAndIndices();
  std::cout << "DSA CPU tests passed\n";
  return 0;
}
