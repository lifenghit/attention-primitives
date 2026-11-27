#include "attention/kda.hpp"
#include <iostream>

int main()
{
    const float q[4] = {1, 0, 0, 1};
    const float k[4] = {1, 0, 0, 1};
    const float v[4] = {1, 2, 3, 4};
    const float g[4] = {-0.1F, -0.2F, -0.3F, -0.4F};
    const float beta[2] = {0.5F, 0.25F};
    float output[4] = {};
    float state[4] = {};
    attention::linear::KDAParams params;
    params.Q = q; params.K = k; params.Values = v; params.g = g; params.beta = beta;
    params.Output = output; params.FinalState = state;
    params.BatchSize = 1; params.SeqLen = 2; params.QKHeads = 1; params.ValueHeads = 1;
    params.KDim = 2; params.VDim = 2;
    if (!attention::linear::kdaRecurrence(params)) return 1;
    std::cout << "KDA output: " << output[0] << ' ' << output[1] << ' ' << output[2] << ' ' << output[3] << '\n';
    return 0;
}
