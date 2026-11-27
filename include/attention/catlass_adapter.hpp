#ifndef ATTENTION_CATLASS_ADAPTER_HPP
#define ATTENTION_CATLASS_ADAPTER_HPP

#include <cstddef>
#include <cstdint>

namespace attention {

struct GemmShape {
    std::uint32_t m;
    std::uint32_t n;
    std::uint32_t k;
};

struct ChunkGemmShapes {
    GemmShape bc;      // C^T[L,N] @ B[N,L] -> M[L,L]
    GemmShape stateC;  // StateSnapshot[P,N] @ C[N,L] -> YStateRaw[P,L]
    GemmShape wx;      // X[P,L] @ W^T[L,L] -> YIntra[P,L]
    GemmShape xb;      // X[P,L] @ BTilde^T[L,N] -> StateChunk[P,N]
};

constexpr ChunkGemmShapes MakeChunkGemmShapes(
    std::uint32_t p, std::uint32_t n, std::uint32_t l) noexcept
{
    return {{l, l, n}, {p, l, n}, {p, l, l}, {p, n, l}};
}

struct ChunkScanHostReferenceParams {
    const float* x;             // [P,L], row-major
    const float* b;             // [N,L], row-major
    const float* c;             // [N,L], row-major
    const float* delta;         // [L]
    const float* cumsum;        // [L], inclusive A * delta prefix sum
    const float* stateIn;       // [P,N], row-major, authoritative FP32 state
    float* yIntra;              // [P,L], row-major
    float* yState;              // [P,L], row-major
    float* stateOut;            // [P,N], row-major, authoritative FP32 state
    std::uint32_t p;
    std::uint32_t n;
    std::uint32_t l;

    constexpr ChunkGemmShapes GemmShapes() const noexcept
    {
        return MakeChunkGemmShapes(p, n, l);
    }
};

} // namespace attention

#ifdef ATTENTION_ENABLE_CATLASS_DEVICE_ADAPTER

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

namespace attention::catlass {

using ArchTag = Catlass::Arch::AtlasA2;
using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;
using L1TileShape = Catlass::GemmShape<128, 256, 256>;
using L0TileShape = Catlass::GemmShape<128, 256, 64>;
using BlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using BlockEpilogue = void;
using Accumulator = float;

template <class LayoutA, class LayoutB>
struct BasicFp16Gemm {
    using AType = Catlass::Gemm::GemmType<half, LayoutA>;
    using BType = Catlass::Gemm::GemmType<half, LayoutB>;
    using CType = Catlass::Gemm::GemmType<Accumulator, Catlass::layout::RowMajor>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using Kernel = Catlass::Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
    using Arguments = typename Kernel::Arguments;
    static __aicore__ void Run(const Arguments& arguments)
    {
        const auto params = Kernel::ToUnderlyingArguments(arguments, nullptr);
        Kernel{}(params);
    }
};

using BCGemm = BasicFp16Gemm<Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>;
using StateCGemm = BasicFp16Gemm<Catlass::layout::RowMajor, Catlass::layout::RowMajor>;
using WXGemm = BasicFp16Gemm<Catlass::layout::RowMajor, Catlass::layout::ColumnMajor>;
using XBGemm = BasicFp16Gemm<Catlass::layout::RowMajor, Catlass::layout::ColumnMajor>;

struct ChunkGemmAddresses {
    GM_ADDR x;
    GM_ADDR b;
    GM_ADDR c;
    GM_ADDR w;
    GM_ADDR bTilde;
    GM_ADDR stateSnapshot;  // FP16 input copy only; never the FP32 recurrence state
    GM_ADDR m;
    GM_ADDR yStateRaw;
    GM_ADDR yIntra;
    GM_ADDR stateChunk;
};

struct ChunkGemmArguments {
    BCGemm::Arguments bc;
    StateCGemm::Arguments stateC;
    WXGemm::Arguments wx;
    XBGemm::Arguments xb;
};

inline ChunkGemmArguments MakeChunkGemmArguments(
    std::uint32_t p, std::uint32_t n, std::uint32_t l,
    const ChunkGemmAddresses& addresses) noexcept
{
    const auto shapes = MakeChunkGemmShapes(p, n, l);
    return {
        {Catlass::GemmCoord(shapes.bc.m, shapes.bc.n, shapes.bc.k),
            addresses.c, addresses.b, addresses.m},
        {Catlass::GemmCoord(shapes.stateC.m, shapes.stateC.n, shapes.stateC.k),
            addresses.stateSnapshot, addresses.c, addresses.yStateRaw},
        {Catlass::GemmCoord(shapes.wx.m, shapes.wx.n, shapes.wx.k),
            addresses.x, addresses.w, addresses.yIntra},
        {Catlass::GemmCoord(shapes.xb.m, shapes.xb.n, shapes.xb.k),
            addresses.x, addresses.bTilde, addresses.stateChunk},
    };
}

} // namespace attention::catlass

#endif // ATTENTION_ENABLE_CATLASS_DEVICE_ADAPTER

#endif // ATTENTION_CATLASS_ADAPTER_HPP
