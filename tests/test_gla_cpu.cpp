#include "attention/gla.hpp"

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

struct Case {
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> gk;
  std::vector<float> initial;
};

Case makeCase() {
  return {
      {0.5F, -1.0F, 0.25F, 1.5F, -0.75F, 0.5F, 1.0F, 0.25F,
       -0.5F, 0.75F, 1.25F, -1.0F},
      {1.0F, 0.5F, -0.25F, 1.25F, 0.75F, -0.5F, 0.5F, 1.0F,
       0.25F, -0.75F, 1.5F, 0.5F},
      {0.5F, 1.0F, -0.25F, 0.75F, 1.5F, -0.5F, 0.25F, 1.25F,
       -1.0F, 0.5F, 0.75F, -0.25F},
      {-0.1F, -0.4F, -0.2F, -0.3F, -0.7F, -0.05F, -0.25F, -0.6F,
       -0.15F, -0.5F, -0.35F, -0.45F},
      {0.25F, -0.5F, 0.75F, 1.0F, -0.75F, 0.5F, 0.2F, -0.4F},
  };
}

void reference(const Case& c, std::vector<float>& output,
               std::vector<float>& state) {
  constexpr std::size_t D = 2;
  constexpr std::size_t DV = 2;
  const float scale = 1.0F / std::sqrt(static_cast<float>(D));
  for (std::size_t b = 0; b < 1; ++b) {
    for (std::size_t h = 0; h < 2; ++h) {
      for (std::size_t t = 0; t < 3; ++t) {
        const std::size_t token = (t * 2 + h);
        for (std::size_t i = 0; i < D; ++i) {
          for (std::size_t j = 0; j < DV; ++j) {
            const std::size_t s = h * 4 + i * DV + j;
            state[s] = state[s] * std::exp(c.gk[token * D + i]) +
                       c.k[token * D + i] * c.v[token * DV + j];
          }
        }
        for (std::size_t j = 0; j < DV; ++j) {
          float value = 0.0F;
          for (std::size_t i = 0; i < D; ++i) {
            value += c.q[token * D + i] * scale * state[h * 4 + i * DV + j];
          }
          output[token * DV + j] = value;
        }
      }
    }
  }
}

void testPerDimensionDecayInitialAndFinalState() {
  Case c = makeCase();
  std::vector<float> expectedOutput(c.v.size());
  std::vector<float> expectedState = c.initial;
  reference(c, expectedOutput, expectedState);

  std::vector<float> output(c.v.size());
  std::vector<float> finalState(c.initial.size());
  attention::linear::GLAParams params;
  params.Q = c.q.data();
  params.K = c.k.data();
  params.V = c.v.data();
  params.gk = c.gk.data();
  params.InitialState = c.initial.data();
  params.B = 1;
  params.T = 3;
  params.H = 2;
  params.D = 2;
  params.DV = 2;
  params.Output = output.data();
  params.FinalState = finalState.data();
  if (!attention::linear::glaRecurrence(params)) {
    std::exit(1);
  }
  for (std::size_t i = 0; i < output.size(); ++i) {
    expectNear(output[i], expectedOutput[i]);
  }
  for (std::size_t i = 0; i < finalState.size(); ++i) {
    expectNear(finalState[i], expectedState[i]);
  }
}

void testZeroInitialStateWithoutFinalState() {
  Case c = makeCase();
  std::fill(c.initial.begin(), c.initial.end(), 0.0F);
  std::vector<float> expectedOutput(c.v.size());
  std::vector<float> expectedState = c.initial;
  reference(c, expectedOutput, expectedState);

  std::vector<float> output(c.v.size());
  attention::linear::GLAParams params;
  params.Q = c.q.data();
  params.K = c.k.data();
  params.V = c.v.data();
  params.gk = c.gk.data();
  params.B = 1;
  params.T = 3;
  params.H = 2;
  params.D = 2;
  params.DV = 2;
  params.Output = output.data();
  if (!attention::linear::glaRecurrence(params)) {
    std::exit(1);
  }
  for (std::size_t i = 0; i < output.size(); ++i) {
    expectNear(output[i], expectedOutput[i]);
  }
}

void testSingleTokenAndInvalidArguments() {
  const float q[] = {2.0F};
  const float k[] = {3.0F};
  const float v[] = {4.0F};
  const float gk[] = {-0.5F};
  const float initial[] = {5.0F};
  float output = 0.0F;
  float finalState = 0.0F;
  attention::linear::GLAParams params;
  params.Q = q;
  params.K = k;
  params.V = v;
  params.gk = gk;
  params.InitialState = initial;
  params.B = params.T = params.H = params.D = params.DV = 1;
  params.Output = &output;
  params.FinalState = &finalState;
  if (!attention::linear::glaRecurrence(params)) {
    std::exit(1);
  }
  expectNear(finalState, 5.0F * std::exp(-0.5F) + 12.0F);
  expectNear(output, 2.0F * finalState);

  if (attention::linear::glaRecurrence(attention::linear::GLAParams{})) {
    std::exit(1);
  }
}

}  // namespace

int main() {
  testPerDimensionDecayInitialAndFinalState();
  testZeroInitialStateWithoutFinalState();
  testSingleTokenAndInvalidArguments();
  std::cout << "GLA CPU tests passed\n";
  return 0;
}
