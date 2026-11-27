#include "attention/qsa.hpp"
#include <iostream>

int main()
{
    const float q[2] = {1, 1};
    const float k[2] = {1, 1};
    const float v[2] = {2, 4};
    const float index_q[2] = {1, 1};
    const float index_k[2] = {1, 1};
    float output[2] = {};
    float scores[2] = {};
    int visible[2] = {};
    int selected[2] = {};
    int physical[2] = {};
    int counts[2] = {};
    attention::QSAParams params;
    params.q = q; params.k = k; params.v = v;
    params.index_q = index_q; params.index_k = index_k;
    params.output = output; params.scores = scores; params.visible_blocks = visible;
    params.selected_indices = selected; params.physical_blocks = physical; params.selected_counts = counts;
    params.sequence_length = 2; params.heads = 1; params.kv_heads = 1; params.index_heads = 1;
    params.head_dim = 1; params.value_dim = 1; params.index_dim = 1;
    params.physical_block_count = 1; params.compress_ratio = 2; params.top_k = 1;
    params.score_divisor = 1.0F; params.query_scale = 1.0F;
    if (attention::qsa(params) != attention::QSAStatus::kOk) return 1;
    std::cout << "QSA output: " << output[0] << ' ' << output[1] << '\n';
    return 0;
}
