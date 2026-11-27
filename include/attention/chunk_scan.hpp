#pragma once

#include <cmath>
#include <cstddef>

namespace attention {
namespace linear {

struct ChunkScanParams {
  const float* X = nullptr;
  const float* B = nullptr;
  const float* C = nullptr;
  const float* delta = nullptr;
  const float* StateIn = nullptr;
  float A = 1.0F;
  std::size_t P = 0;
  std::size_t N = 0;
  std::size_t L = 0;
  float* cumsum = nullptr;
  float* W = nullptr;
  float* YIntra = nullptr;
  float* YState = nullptr;
};

// Row-major layouts: X [P,L], B/C [N,L], StateIn [P,N].
// Outputs: cumsum [L], W [L,L], YIntra/YState [P,L]. All arithmetic is FP32.
inline bool chunkScan(const ChunkScanParams& params) noexcept {
  if (params.P == 0 || params.N == 0 || params.L == 0 || params.X == nullptr ||
      params.B == nullptr || params.C == nullptr || params.delta == nullptr ||
      params.StateIn == nullptr || params.cumsum == nullptr ||
      params.W == nullptr || params.YIntra == nullptr ||
      params.YState == nullptr) {
    return false;
  }

  float cumulative = 0.0F;
  for (std::size_t j = 0; j < params.L; ++j) {
    cumulative += params.A * params.delta[j];
    params.cumsum[j] = cumulative;
  }

  for (std::size_t i = 0; i < params.L; ++i) {
    for (std::size_t j = 0; j < params.L; ++j) {
      float value = 0.0F;
      if (j <= i) {
        float m = 0.0F;
        for (std::size_t n = 0; n < params.N; ++n) {
          m += params.C[n * params.L + i] * params.B[n * params.L + j];
        }
        value = m * std::exp(params.cumsum[i] - params.cumsum[j]) *
                params.delta[j];
      }
      params.W[i * params.L + j] = value;
    }
  }

  for (std::size_t p = 0; p < params.P; ++p) {
    for (std::size_t i = 0; i < params.L; ++i) {
      float intra = 0.0F;
      for (std::size_t j = 0; j <= i; ++j) {
        intra += params.X[p * params.L + j] * params.W[i * params.L + j];
      }
      params.YIntra[p * params.L + i] = intra;

      float stateReadout = 0.0F;
      for (std::size_t n = 0; n < params.N; ++n) {
        stateReadout += params.StateIn[p * params.N + n] *
                        params.C[n * params.L + i];
      }
      params.YState[p * params.L + i] =
          std::exp(params.cumsum[i]) * stateReadout;
    }
  }
  return true;
}

}  // namespace linear
}  // namespace attention
