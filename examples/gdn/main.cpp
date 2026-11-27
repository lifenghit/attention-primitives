#include "attention/gdn.hpp"
#include <iostream>

int main()
{
    const float q[4] = {1, 0, 0, 1};
    const float k[4] = {1, 0, 0, 1};
    const float v[4] = {1, 2, 3, 4};
    const float g[2] = {-0.1F, -0.2F};
    const float beta[2] = {0.5F, 0.25F};
    float output[4] = {};
    float state[4] = {};
    attention::linear::GdnParams params;
    params.q = q; params.k = k; params.v = v; params.g = g; params.beta = beta;
    params.o = output; params.finalState = state; params.outputFinalState = true;
    params.B = 1; params.T = 2; params.H = 1; params.K = 2; params.V = 2;
    if (!attention::linear::gdn(params)) return 1;
    std::cout << "GDN output: " << output[0] << ' ' << output[1] << ' ' << output[2] << ' ' << output[3] << '\n';
    return 0;
}
