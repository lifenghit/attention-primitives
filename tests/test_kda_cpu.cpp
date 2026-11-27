#include "attention/kda.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void expectNear(float actual, float expected, float tolerance = 1.0e-5F) {
  const float scale = std::max({1.0F, std::fabs(actual), std::fabs(expected)});
  if (std::fabs(actual - expected) > tolerance * scale) {
    std::cerr << "expected " << expected << ", got " << actual << '\n';
    std::exit(1);
  }
}

void testGvaMultiTokenAndNonzeroState() {
  constexpr std::size_t B = 1;
  constexpr std::size_t T = 3;
  constexpr std::size_t H = 2;
  constexpr std::size_t HV = 4;
  constexpr std::size_t K = 2;
  constexpr std::size_t V = 2;
  const std::vector<float> q{
      0.5F, -0.25F, 0.75F, 0.5F,
      0.25F, 0.5F, -0.5F, 0.25F,
      1.0F, -0.5F, 0.25F, 0.75F,
  };
  const std::vector<float> k{
      0.5F, 0.25F, -0.25F, 0.75F,
      0.75F, -0.5F, 0.5F, 0.25F,
      -0.25F, 0.5F, 0.25F, -0.75F,
  };
  std::vector<float> v(T * HV * V);
  std::vector<float> g(T * HV * K);
  std::vector<float> beta(T * HV);
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = 0.05F * static_cast<float>(static_cast<int>(i * 7 % 13) - 6);
  }
  for (std::size_t i = 0; i < g.size(); ++i) {
    g[i] = -0.03F * static_cast<float>(1 + i % 5);
  }
  for (std::size_t i = 0; i < beta.size(); ++i) {
    beta[i] = 0.2F + 0.05F * static_cast<float>(i % 4);
  }
  std::vector<float> initial(B * HV * K * V);
  for (std::size_t i = 0; i < initial.size(); ++i) {
    initial[i] = 0.02F * static_cast<float>(static_cast<int>(i * 5 % 11) - 5);
  }
  std::vector<float> output(v.size());
  std::vector<float> finalState(initial.size());
  attention::linear::KDAParams params;
  params.Q = q.data();
  params.K = k.data();
  params.Values = v.data();
  params.g = g.data();
  params.beta = beta.data();
  params.InitialState = initial.data();
  params.Output = output.data();
  params.FinalState = finalState.data();
  params.BatchSize = B;
  params.SeqLen = T;
  params.QKHeads = H;
  params.ValueHeads = HV;
  params.KDim = K;
  params.VDim = V;
  if (!attention::linear::kdaRecurrence(params)) {
    std::exit(1);
  }

  std::vector<float> expectedState = initial;
  std::vector<float> expectedOutput(v.size());
  const float scale = 1.0F / std::sqrt(static_cast<float>(K));
  for (std::size_t t = 0; t < T; ++t) {
    for (std::size_t hv = 0; hv < HV; ++hv) {
      const std::size_t h = hv / (HV / H);
      for (std::size_t d = 0; d < K; ++d) {
        for (std::size_t n = 0; n < V; ++n) {
          expectedState[(hv * K + d) * V + n] *=
              std::exp(g[(t * HV + hv) * K + d]);
        }
      }
      for (std::size_t n = 0; n < V; ++n) {
        float prediction = 0.0F;
        for (std::size_t d = 0; d < K; ++d) {
          prediction += k[(t * H + h) * K + d] *
                        expectedState[(hv * K + d) * V + n];
        }
        const float delta = v[(t * HV + hv) * V + n] - prediction;
        for (std::size_t d = 0; d < K; ++d) {
          expectedState[(hv * K + d) * V + n] +=
              beta[t * HV + hv] * k[(t * H + h) * K + d] * delta;
        }
        for (std::size_t d = 0; d < K; ++d) {
          expectedOutput[(t * HV + hv) * V + n] +=
              scale * q[(t * H + h) * K + d] *
              expectedState[(hv * K + d) * V + n];
        }
      }
    }
  }
  for (std::size_t i = 0; i < output.size(); ++i) {
    expectNear(output[i], expectedOutput[i]);
  }
  for (std::size_t i = 0; i < finalState.size(); ++i) {
    expectNear(finalState[i], expectedState[i]);
  }
  expectNear(output[0], -0.0582512468F);
  expectNear(output[1], 0.0047803270F);
  expectNear(finalState[0], -0.1037526056F);
}

void testZeroInitialStateAndCustomScale() {
  const std::vector<float> value{2.0F};
  const std::vector<float> gate{-0.25F};
  const std::vector<float> beta{0.5F};
  float output = 0.0F;
  float state = 0.0F;
  attention::linear::KDAParams params;
  params.Q = value.data();
  params.K = value.data();
  params.Values = value.data();
  params.g = gate.data();
  params.beta = beta.data();
  params.Output = &output;
  params.FinalState = &state;
  params.BatchSize = 1;
  params.SeqLen = 1;
  params.QKHeads = 1;
  params.ValueHeads = 1;
  params.KDim = 1;
  params.VDim = 1;
  params.Scale = 0.5F;
  if (!attention::linear::kdaRecurrence(params)) {
    std::exit(1);
  }
  expectNear(state, 2.0F);
  expectNear(output, 2.0F);
}

void testInvalidArguments() {
  attention::linear::KDAParams params;
  if (attention::linear::kdaRecurrence(params)) {
    std::exit(1);
  }
}

}  // namespace

int main() {
  testGvaMultiTokenAndNonzeroState();
  testZeroInitialStateAndCustomScale();
  testInvalidArguments();
  std::cout << "KDA CPU tests passed\n";
  return 0;
}
