#pragma once

#include <cmath>
#include <cstddef>

namespace attention {
namespace linear {

struct KDAParams {
  const float* Q = nullptr;
  const float* K = nullptr;
  const float* Values = nullptr;
  const float* g = nullptr;
  const float* beta = nullptr;
  const float* InitialState = nullptr;
  float* Output = nullptr;
  float* FinalState = nullptr;
  std::size_t BatchSize = 0;
  std::size_t SeqLen = 0;
  std::size_t QKHeads = 0;
  std::size_t ValueHeads = 0;
  std::size_t KDim = 0;
  std::size_t VDim = 0;
  float Scale = -1.0F;
};

// Q/K [B,T,H,K], Values [B,T,HV,V], g [B,T,HV,K], beta [B,T,HV].
// InitialState/FinalState [B,HV,K,V]; a null InitialState means zero.
inline bool kdaRecurrence(const KDAParams& params) noexcept {
  if (params.BatchSize == 0 || params.SeqLen == 0 || params.QKHeads == 0 ||
      params.ValueHeads == 0 || params.ValueHeads % params.QKHeads != 0 ||
      params.KDim == 0 || params.VDim == 0 || params.Q == nullptr ||
      params.K == nullptr || params.Values == nullptr || params.g == nullptr ||
      params.beta == nullptr || params.Output == nullptr ||
      params.FinalState == nullptr) {
    return false;
  }

  const std::size_t groupSize = params.ValueHeads / params.QKHeads;
  const float scale = params.Scale < 0.0F
                          ? 1.0F / std::sqrt(static_cast<float>(params.KDim))
                          : params.Scale;
  for (std::size_t b = 0; b < params.BatchSize; ++b) {
    for (std::size_t hv = 0; hv < params.ValueHeads; ++hv) {
      for (std::size_t d = 0; d < params.KDim; ++d) {
        for (std::size_t n = 0; n < params.VDim; ++n) {
          const std::size_t stateIndex =
              ((b * params.ValueHeads + hv) * params.KDim + d) * params.VDim + n;
          params.FinalState[stateIndex] = params.InitialState == nullptr
                                              ? 0.0F
                                              : params.InitialState[stateIndex];
        }
      }
    }
    for (std::size_t t = 0; t < params.SeqLen; ++t) {
      for (std::size_t hv = 0; hv < params.ValueHeads; ++hv) {
        const std::size_t h = hv / groupSize;
        const std::size_t qkBase =
            ((b * params.SeqLen + t) * params.QKHeads + h) * params.KDim;
        const std::size_t valueBase =
            ((b * params.SeqLen + t) * params.ValueHeads + hv) * params.VDim;
        const std::size_t gateBase =
            ((b * params.SeqLen + t) * params.ValueHeads + hv) * params.KDim;
        const std::size_t stateBase = (b * params.ValueHeads + hv) *
                                      params.KDim * params.VDim;
        const float beta =
            params.beta[(b * params.SeqLen + t) * params.ValueHeads + hv];
        for (std::size_t d = 0; d < params.KDim; ++d) {
          const float decay = std::exp(params.g[gateBase + d]);
          for (std::size_t n = 0; n < params.VDim; ++n) {
            params.FinalState[stateBase + d * params.VDim + n] *= decay;
          }
        }
        for (std::size_t n = 0; n < params.VDim; ++n) {
          float prediction = 0.0F;
          for (std::size_t d = 0; d < params.KDim; ++d) {
            prediction += params.K[qkBase + d] *
                          params.FinalState[stateBase + d * params.VDim + n];
          }
          const float delta = params.Values[valueBase + n] - prediction;
          for (std::size_t d = 0; d < params.KDim; ++d) {
            params.FinalState[stateBase + d * params.VDim + n] +=
                beta * params.K[qkBase + d] * delta;
          }
          float result = 0.0F;
          for (std::size_t d = 0; d < params.KDim; ++d) {
            result += scale * params.Q[qkBase + d] *
                      params.FinalState[stateBase + d * params.VDim + n];
          }
          params.Output[valueBase + n] = result;
        }
      }
    }
  }
  return true;
}

}  // namespace linear
}  // namespace attention
