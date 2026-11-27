#include "attention/dsa.hpp"
#include <iostream>

int main()
{
    const float q[2] = {1, 1};
    const float k[2] = {1, 1};
    const float v[2] = {2, 4};
    const float q_idx[2] = {1, 1};
    const float k_idx[2] = {1, 1};
    const float w_idx[2] = {1, 1};
    float output[2] = {};
    int selected[2] = {};
    attention::DsaParams params;
    params.q = q; params.k = k; params.v = v;
    params.q_idx = q_idx; params.k_idx = k_idx; params.w_idx = w_idx;
    params.output = output; params.selected_indices = selected;
    params.batch = 1; params.sequence_length = 2;
    params.query_heads = 1; params.kv_heads = 1; params.head_dim = 1; params.value_dim = 1;
    params.index_heads = 1; params.index_dim = 1; params.top_k = 1;
    if (attention::dsa(params) != attention::DsaStatus::kOk) return 1;
    std::cout << "DSA output: " << output[0] << ' ' << output[1] << '\n';
    return 0;
}
