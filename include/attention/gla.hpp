#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace attention {
namespace linear {

struct GLAParams {
  const float* Q = nullptr;
  const float* K = nullptr;
  const float* V = nullptr;
  const float* gk = nullptr;
  const float* InitialState = nullptr;
  std::size_t B = 0;
  std::size_t T = 0;
  std::size_t H = 0;
  std::size_t D = 0;
  std::size_t DV = 0;
  float* Output = nullptr;
  float* FinalState = nullptr;
};

// Q/K/gk [B,T,H,D], V/Output [B,T,H,DV], state [B,H,D,DV], all row-major.
// InitialState is optional; FinalState is written only when non-null.
inline bool glaRecurrence(const GLAParams& params) {
  if (params.B == 0 || params.T == 0 || params.H == 0 || params.D == 0 ||
      params.DV == 0 || params.Q == nullptr || params.K == nullptr ||
      params.V == nullptr || params.gk == nullptr || params.Output == nullptr) {
    return false;
  }

  const float scale = 1.0F / std::sqrt(static_cast<float>(params.D));
  const std::size_t stateSize = params.D * params.DV;
  const std::size_t batchStateStride = params.H * stateSize;
  std::vector<float> localState;
  if (params.FinalState == nullptr) {
    localState.resize(stateSize);
  }

  for (std::size_t b = 0; b < params.B; ++b) {
    for (std::size_t h = 0; h < params.H; ++h) {
      const std::size_t stateOffset = b * batchStateStride + h * stateSize;
      float* state = params.FinalState == nullptr
                         ? localState.data()
                         : params.FinalState + stateOffset;
      for (std::size_t i = 0; i < stateSize; ++i) {
        state[i] = params.InitialState == nullptr
                       ? 0.0F
                       : params.InitialState[stateOffset + i];
      }

      for (std::size_t t = 0; t < params.T; ++t) {
        const std::size_t tokenOffset = (b * params.T + t) * params.H + h;
        const std::size_t keyOffset = tokenOffset * params.D;
        const std::size_t valueOffset = tokenOffset * params.DV;
        for (std::size_t i = 0; i < params.D; ++i) {
          const float decay = std::exp(params.gk[keyOffset + i]);
          const float key = params.K[keyOffset + i];
          for (std::size_t j = 0; j < params.DV; ++j) {
            const std::size_t stateIndex = i * params.DV + j;
            state[stateIndex] = state[stateIndex] * decay +
                                key * params.V[valueOffset + j];
          }
        }
        for (std::size_t j = 0; j < params.DV; ++j) {
          float value = 0.0F;
          for (std::size_t i = 0; i < params.D; ++i) {
            value += params.Q[keyOffset + i] * scale * state[i * params.DV + j];
          }
          params.Output[valueOffset + j] = value;
        }
      }
    }
  }
  return true;
}

}  // namespace linear
}  // namespace attention
