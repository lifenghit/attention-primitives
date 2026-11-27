#include "attention/msa.hpp"
#include <iostream>

int main()
{
    const float q[2] = {1, 1};
    const float k[2] = {1, 1};
    const float v[2] = {2, 4};
    const float q_idx[2] = {1, 1};
    const float k_idx[2] = {1, 1};
    float output[2] = {};
    int selected[2] = {};
    int counts[2] = {};
    attention::MSAParams params;
    params.q = q; params.k = k; params.v = v; params.q_idx = q_idx; params.k_idx = k_idx;
    params.output = output; params.indices = selected; params.counts = counts;
    params.sequence_length = 2; params.q_heads = 1; params.kv_heads = 1;
    params.d = 1; params.v_dim = 1; params.index_dim = 1;
    params.block_size = 1; params.topk_blocks = 1;
    if (attention::msa(params) != attention::MSAStatus::kOk) return 1;
    std::cout << "MSA output: " << output[0] << ' ' << output[1] << '\n';
    return 0;
}
