#include "attention/chunk_scan.hpp"
#include "attention/state_update.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

using attention::linear::ChunkScanParams;
using attention::linear::StateUpdateParams;

void expectNear(float actual, float expected, float tolerance = 2.0e-5F) {
  const float scale = std::max({1.0F, std::fabs(actual), std::fabs(expected)});
  if (std::fabs(actual - expected) > tolerance * scale) {
    std::cerr << "expected " << expected << ", got " << actual << '\n';
    std::exit(1);
  }
}

struct ChunkResult {
  std::vector<float> cumsum;
  std::vector<float> W;
  std::vector<float> YIntra;
  std::vector<float> YState;
  std::vector<float> StateOut;
};

ChunkResult runChunk(const std::vector<float>& X, const std::vector<float>& B,
                     const std::vector<float>& C,
                     const std::vector<float>& delta, float A, std::size_t P,
                     std::size_t N, const std::vector<float>& StateIn) {
  const std::size_t L = delta.size();
  ChunkResult result{std::vector<float>(L), std::vector<float>(L * L),
                     std::vector<float>(P * L), std::vector<float>(P * L),
                     std::vector<float>(P * N)};
  ChunkScanParams scan;
  scan.X = X.data();
  scan.B = B.data();
  scan.C = C.data();
  scan.delta = delta.data();
  scan.StateIn = StateIn.data();
  scan.A = A;
  scan.P = P;
  scan.N = N;
  scan.L = L;
  scan.cumsum = result.cumsum.data();
  scan.W = result.W.data();
  scan.YIntra = result.YIntra.data();
  scan.YState = result.YState.data();
  if (!attention::linear::chunkScan(scan)) {
    std::exit(1);
  }

  StateUpdateParams update;
  update.X = X.data();
  update.B = B.data();
  update.delta = delta.data();
  update.cumsum = result.cumsum.data();
  update.StateIn = StateIn.data();
  update.P = P;
  update.N = N;
  update.L = L;
  update.StateOut = result.StateOut.data();
  if (!attention::linear::stateUpdate(update)) {
    std::exit(1);
  }
  return result;
}

void testLengthOneAndNonzeroState() {
  const ChunkResult r = runChunk({2.0F}, {3.0F}, {4.0F}, {0.5F}, 2.0F, 1, 1,
                                 {5.0F});
  expectNear(r.cumsum[0], 1.0F);
  expectNear(r.W[0], 6.0F);
  expectNear(r.YIntra[0], 12.0F);
  expectNear(r.YState[0], 20.0F * std::exp(1.0F));
  expectNear(r.StateOut[0], 3.0F + 5.0F * std::exp(1.0F));
}

void testMultiChunkAndTailAgainstTokenRecurrence() {
  constexpr std::size_t P = 2;
  constexpr std::size_t N = 2;
  constexpr float A = -0.75F;
  const std::vector<float> X{0.5F, -1.0F, 1.5F, 2.0F, 0.25F, -0.5F};
  const std::vector<float> B{1.0F, 0.5F, -1.0F, -0.25F, 2.0F, 1.0F};
  const std::vector<float> C{0.75F, -0.5F, 1.25F, 1.0F, -0.75F, 0.5F};
  const std::vector<float> delta{0.25F, 0.5F, 0.125F};
  std::vector<float> state{0.25F, -0.5F, 1.0F, 2.0F};

  const ChunkResult first =
      runChunk({X[0], X[1], X[3], X[4]}, {B[0], B[1], B[3], B[4]},
               {C[0], C[1], C[3], C[4]}, {delta[0], delta[1]}, A, P, N,
               state);
  expectNear(first.cumsum[0], A * delta[0]);
  expectNear(first.cumsum[1], A * (delta[0] + delta[1]));
  expectNear(first.W[1], 0.0F);

  const ChunkResult tail =
      runChunk({X[2], X[5]}, {B[2], B[5]}, {C[2], C[5]}, {delta[2]}, A, P,
               N, first.StateOut);

  std::vector<float> expectedState = state;
  std::vector<float> expectedY(P * delta.size());
  for (std::size_t t = 0; t < delta.size(); ++t) {
    const float decay = std::exp(A * delta[t]);
    for (std::size_t p = 0; p < P; ++p) {
      for (std::size_t n = 0; n < N; ++n) {
        expectedState[p * N + n] =
            decay * expectedState[p * N + n] +
            X[p * delta.size() + t] * B[n * delta.size() + t] * delta[t];
      }
      for (std::size_t n = 0; n < N; ++n) {
        expectedY[p * delta.size() + t] +=
            C[n * delta.size() + t] * expectedState[p * N + n];
      }
    }
  }

  for (std::size_t p = 0; p < P; ++p) {
    for (std::size_t t = 0; t < 2; ++t) {
      expectNear(first.YIntra[p * 2 + t] + first.YState[p * 2 + t],
                 expectedY[p * 3 + t]);
    }
    expectNear(tail.YIntra[p] + tail.YState[p], expectedY[p * 3 + 2]);
    for (std::size_t n = 0; n < N; ++n) {
      expectNear(tail.StateOut[p * N + n], expectedState[p * N + n]);
    }
  }
}

void testZeroStateAndUpperTriangle() {
  const ChunkResult r =
      runChunk({1.0F, 2.0F}, {1.0F, 1.0F}, {1.0F, 1.0F}, {1.0F, 1.0F},
               0.0F, 1, 1, {0.0F});
  expectNear(r.W[0], 1.0F);
  expectNear(r.W[1], 0.0F);
  expectNear(r.W[2], 1.0F);
  expectNear(r.W[3], 1.0F);
  expectNear(r.YIntra[0], 1.0F);
  expectNear(r.YIntra[1], 3.0F);
  expectNear(r.YState[0], 0.0F);
  expectNear(r.YState[1], 0.0F);
  expectNear(r.StateOut[0], 3.0F);
}

void testInvalidShapesRejected() {
  ChunkScanParams scan;
  if (attention::linear::chunkScan(scan)) {
    std::exit(1);
  }
  StateUpdateParams update;
  if (attention::linear::stateUpdate(update)) {
    std::exit(1);
  }
}

}  // namespace

int main() {
  testLengthOneAndNonzeroState();
  testMultiChunkAndTailAgainstTokenRecurrence();
  testZeroStateAndUpperTriangle();
  testInvalidShapesRejected();
  std::cout << "linear CPU tests passed\n";
  return 0;
}
