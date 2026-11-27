#include "attention/gdn.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expectNear(float actual, double expected, double tolerance = 1.0e-5) {
  const double scale = std::max({1.0, std::fabs(static_cast<double>(actual)),
                                 std::fabs(expected)});
  if (std::fabs(static_cast<double>(actual) - expected) > tolerance * scale) {
    std::cerr << "expected " << expected << ", got " << actual << '\n';
    ++failures;
  }
}

void expectTrue(bool condition) {
  if (!condition) {
    std::cerr << "assertion failed\n";
    ++failures;
  }
}

struct ReferenceResult {
  std::vector<double> output;
  std::vector<double> state;
};

ReferenceResult reference(const std::vector<float>& q,
                          const std::vector<float>& k,
                          const std::vector<float>& v,
                          const std::vector<float>& beta,
                          const std::vector<float>& g,
                          const std::vector<float>& initialState,
                          std::size_t B, std::size_t T, std::size_t H,
                          std::size_t K, std::size_t V, double scale) {
  ReferenceResult result{std::vector<double>(B * T * H * V, 0.0),
                         std::vector<double>(initialState.begin(),
                                             initialState.end())};
  for (std::size_t b = 0; b < B; ++b) {
    for (std::size_t h = 0; h < H; ++h) {
      double* state = result.state.data() + (b * H + h) * K * V;
      for (std::size_t t = 0; t < T; ++t) {
        const std::size_t token = (b * T + t) * H + h;
        const double decay = std::exp(static_cast<double>(g[token]));
        for (std::size_t i = 0; i < K; ++i) {
          for (std::size_t j = 0; j < V; ++j) {
            state[i * V + j] *= decay;
          }
        }
        for (std::size_t j = 0; j < V; ++j) {
          double prediction = 0.0;
          for (std::size_t i = 0; i < K; ++i) {
            prediction += static_cast<double>(k[token * K + i]) * state[i * V + j];
          }
          const double delta = static_cast<double>(beta[token]) *
                               (static_cast<double>(v[token * V + j]) - prediction);
          for (std::size_t i = 0; i < K; ++i) {
            state[i * V + j] += static_cast<double>(k[token * K + i]) * delta;
          }
          for (std::size_t i = 0; i < K; ++i) {
            result.output[token * V + j] +=
                static_cast<double>(q[token * K + i]) * scale * state[i * V + j];
          }
        }
      }
    }
  }
  return result;
}

void testSingleTokenWithInitialState() {
  const float logDecay = std::log(0.25F);
  attention::linear::GdnParams params;
  const std::vector<float> q{2.0F};
  const std::vector<float> k{3.0F};
  const std::vector<float> v{10.0F};
  const std::vector<float> beta{0.5F};
  const std::vector<float> g{logDecay};
  const std::vector<float> initial{4.0F};
  std::vector<float> output(1, 0.0F);
  std::vector<float> finalState(1, 0.0F);
  params.q = q.data();
  params.k = k.data();
  params.v = v.data();
  params.beta = beta.data();
  params.g = g.data();
  params.initialState = initial.data();
  params.scale = 1.0F;
  params.B = params.T = params.H = params.K = params.V = 1;
  params.o = output.data();
  params.finalState = finalState.data();
  params.outputFinalState = true;

  expectTrue(attention::linear::gatedDeltaRule(params));
  expectNear(output[0], 23.0);
  expectNear(finalState[0], 11.5);
}

void testBatchedScalarGatesAndRectangularState() {
  constexpr std::size_t B = 2;
  constexpr std::size_t T = 3;
  constexpr std::size_t H = 2;
  constexpr std::size_t K = 2;
  constexpr std::size_t V = 3;
  std::vector<float> q(B * T * H * K);
  std::vector<float> k(B * T * H * K);
  std::vector<float> v(B * T * H * V);
  std::vector<float> beta(B * T * H);
  std::vector<float> g(B * T * H);
  std::vector<float> initial(B * H * K * V);
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.2F;
    k[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.15F;
  }
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.3F;
  }
  for (std::size_t i = 0; i < beta.size(); ++i) {
    beta[i] = 0.2F + 0.1F * static_cast<float>(i % 4);
    g[i] = -0.05F * static_cast<float>(i % 5);
  }
  for (std::size_t i = 0; i < initial.size(); ++i) {
    initial[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.1F;
  }

  std::vector<float> output(B * T * H * V, 0.0F);
  std::vector<float> finalState(B * H * K * V, 0.0F);
  attention::linear::GatedDeltaRuleParams params;
  params.q = q.data();
  params.k = k.data();
  params.v = v.data();
  params.beta = beta.data();
  params.g = g.data();
  params.initialState = initial.data();
  params.scale = 0.75F;
  params.B = B;
  params.T = T;
  params.H = H;
  params.K = K;
  params.V = V;
  params.o = output.data();
  params.finalState = finalState.data();
  params.outputFinalState = true;
  expectTrue(attention::linear::gdn(params));

  const ReferenceResult expected =
      reference(q, k, v, beta, g, initial, B, T, H, K, V, 0.75);
  for (std::size_t i = 0; i < output.size(); ++i) {
    expectNear(output[i], expected.output[i]);
  }
  for (std::size_t i = 0; i < finalState.size(); ++i) {
    expectNear(finalState[i], expected.state[i]);
  }
}

void testDefaultScaleZeroStateAndOptionalFinalState() {
  constexpr std::size_t K = 4;
  constexpr std::size_t V = 2;
  const std::vector<float> q(K, 1.0F);
  const std::vector<float> k{1.0F, 0.0F, 0.5F, -0.5F};
  const std::vector<float> v{2.0F, -1.0F};
  const std::vector<float> beta{0.5F};
  const std::vector<float> g{0.0F};
  const std::vector<float> initial(K * V, 0.0F);
  std::vector<float> output(V, 0.0F);
  attention::linear::GdnParams params;
  params.q = q.data();
  params.k = k.data();
  params.v = v.data();
  params.beta = beta.data();
  params.g = g.data();
  params.B = params.T = params.H = 1;
  params.K = K;
  params.V = V;
  params.o = output.data();
  expectTrue(attention::linear::gatedDeltaRule(params));

  const ReferenceResult expected = reference(
      q, k, v, beta, g, initial, 1, 1, 1, K, V, 1.0 / std::sqrt(4.0));
  for (std::size_t j = 0; j < V; ++j) {
    expectNear(output[j], expected.output[j]);
  }

  params.scale = 0.0F;
  expectTrue(attention::linear::gatedDeltaRule(params));
  for (float value : output) {
    expectNear(value, 0.0);
  }
}

void testInvalidArguments() {
  attention::linear::GdnParams params;
  expectTrue(!attention::linear::gatedDeltaRule(params));
  float value = 0.0F;
  params.q = params.k = params.v = params.beta = params.g = &value;
  params.o = &value;
  params.B = params.T = params.H = params.K = params.V = 1;
  params.outputFinalState = true;
  expectTrue(!attention::linear::gatedDeltaRule(params));
  params.outputFinalState = false;
  expectTrue(attention::linear::gatedDeltaRule(params));
}

}  // namespace

int main() {
  testSingleTokenWithInitialState();
  testBatchedScalarGatesAndRectangularState();
  testDefaultScaleZeroStateAndOptionalFinalState();
  testInvalidArguments();
  if (failures != 0) {
    std::cerr << failures << " GDN assertion(s) failed\n";
    return 1;
  }
  std::cout << "All GDN CPU tests passed\n";
  return 0;
}
