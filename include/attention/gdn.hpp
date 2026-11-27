#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace attention {
namespace linear {

struct GdnParams {
  const float* q = nullptr;
  const float* k = nullptr;
  const float* v = nullptr;
  const float* beta = nullptr;
  const float* g = nullptr;
  const float* initialState = nullptr;
  float scale = std::numeric_limits<float>::quiet_NaN();
  std::size_t B = 0;
  std::size_t T = 0;
  std::size_t H = 0;
  std::size_t K = 0;
  std::size_t V = 0;
  float* o = nullptr;
  float* finalState = nullptr;
  bool outputFinalState = false;
};

using GatedDeltaRuleParams = GdnParams;

// Row-major q/k [B,T,H,K], v/o [B,T,H,V], beta/g [B,T,H], state [B,H,K,V].
// A NaN scale selects K^-0.5. All recurrence arithmetic is FP32.
inline bool gatedDeltaRule(const GdnParams& params) {
  if (params.B == 0 || params.T == 0 || params.H == 0 || params.K == 0 ||
      params.V == 0 || params.q == nullptr || params.k == nullptr ||
      params.v == nullptr || params.beta == nullptr || params.g == nullptr ||
      params.o == nullptr ||
      (params.outputFinalState && params.finalState == nullptr)) {
    return false;
  }

  const std::size_t stateSize = params.B * params.H * params.K * params.V;
  std::vector<float> state(stateSize, 0.0F);
  if (params.initialState != nullptr) {
    std::copy(params.initialState, params.initialState + stateSize, state.begin());
  }
  const float scale = std::isnan(params.scale)
                          ? 1.0F / std::sqrt(static_cast<float>(params.K))
                          : params.scale;

  for (std::size_t b = 0; b < params.B; ++b) {
    for (std::size_t h = 0; h < params.H; ++h) {
      float* headState = state.data() + (b * params.H + h) * params.K * params.V;
      for (std::size_t t = 0; t < params.T; ++t) {
        const std::size_t token = (b * params.T + t) * params.H + h;
        const float* q = params.q + token * params.K;
        const float* k = params.k + token * params.K;
        const float* v = params.v + token * params.V;
        float* o = params.o + token * params.V;
        const float decay = std::exp(params.g[token]);
        for (std::size_t i = 0; i < params.K; ++i) {
          for (std::size_t j = 0; j < params.V; ++j) {
            headState[i * params.V + j] *= decay;
          }
        }
        for (std::size_t j = 0; j < params.V; ++j) {
          float prediction = 0.0F;
          for (std::size_t i = 0; i < params.K; ++i) {
            prediction += k[i] * headState[i * params.V + j];
          }
          const float delta = params.beta[token] * (v[j] - prediction);
          for (std::size_t i = 0; i < params.K; ++i) {
            headState[i * params.V + j] += k[i] * delta;
          }
          float readout = 0.0F;
          for (std::size_t i = 0; i < params.K; ++i) {
            readout += (q[i] * scale) * headState[i * params.V + j];
          }
          o[j] = readout;
        }
      }
    }
  }

  if (params.outputFinalState) {
    std::copy(state.begin(), state.end(), params.finalState);
  }
  return true;
}

inline bool gdn(const GdnParams& params) { return gatedDeltaRule(params); }

}  // namespace linear
}  // namespace attention
