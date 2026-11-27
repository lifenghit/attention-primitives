#pragma once

#include <cmath>
#include <cstddef>

namespace attention {
namespace linear {

struct StateUpdateParams {
  const float* X = nullptr;
  const float* B = nullptr;
  const float* delta = nullptr;
  const float* cumsum = nullptr;
  const float* StateIn = nullptr;
  std::size_t P = 0;
  std::size_t N = 0;
  std::size_t L = 0;
  float* StateOut = nullptr;
};

// Row-major layouts: X [P,L], B [N,L], StateIn/StateOut [P,N].
// StateOut is the FP32 recurrence result; callers may separately round a snapshot.
inline bool stateUpdate(const StateUpdateParams& params) noexcept {
  if (params.P == 0 || params.N == 0 || params.L == 0 || params.X == nullptr ||
      params.B == nullptr || params.delta == nullptr || params.cumsum == nullptr ||
      params.StateIn == nullptr || params.StateOut == nullptr) {
    return false;
  }

  const float finalDecay = std::exp(params.cumsum[params.L - 1]);
  for (std::size_t p = 0; p < params.P; ++p) {
    for (std::size_t n = 0; n < params.N; ++n) {
      float state = finalDecay * params.StateIn[p * params.N + n];
      for (std::size_t j = 0; j < params.L; ++j) {
        const float bTilde =
            params.B[n * params.L + j] * params.delta[j] *
            std::exp(params.cumsum[params.L - 1] - params.cumsum[j]);
        state += params.X[p * params.L + j] * bTilde;
      }
      params.StateOut[p * params.N + n] = state;
    }
  }
  return true;
}

}  // namespace linear
}  // namespace attention
