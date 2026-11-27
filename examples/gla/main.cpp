#include "attention/gla.hpp"
#include <iostream>

int main()
{
    const float q[4] = {1, 0, 0, 1};
    const float k[4] = {1, 0, 0, 1};
    const float v[4] = {1, 2, 3, 4};
    const float gk[4] = {-0.1F, -0.2F, -0.3F, -0.4F};
    float output[4] = {};
    float state[4] = {};
    attention::linear::GLAParams params;
    params.Q = q; params.K = k; params.V = v; params.gk = gk;
    params.Output = output; params.FinalState = state;
    params.B = 1; params.T = 2; params.H = 1; params.D = 2; params.DV = 2;
    if (!attention::linear::glaRecurrence(params)) return 1;
    std::cout << "GLA output: " << output[0] << ' ' << output[1] << ' ' << output[2] << ' ' << output[3] << '\n';
    return 0;
}
