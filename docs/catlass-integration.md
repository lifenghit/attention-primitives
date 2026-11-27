# CATLASS 配合层

## 依据与范围

选型依据是远端 CATLASS 提交 `3a6b66970ea1469fb566eb5f50afa9a469606e92` 的以下真实源码：

- `README.md`、`docs/zh/3_API/gemm_api.md`
- `examples/00_basic_matmul/basic_matmul.cpp`
- `examples/40_flash_attention_infer_tla/{README.md,fai_kernel.cpp,kernel_common.hpp}`
- `examples/108_ascend950_mamba_ssd/ssd_design.md`

本层只提供 host 数值契约、四个 GEMM 的组件别名和参数封装，不实现完整 FlashAttention 或 Mamba SSD kernel。`40_flash_attention_infer_tla` 仅用于确认 paged KV、GQA 和 QK/PV 的组件边界；当前公开 `BlockScoreParams` 是均值池化后的块分数，不能直接调用该完整 FA kernel。`108_ascend950_mamba_ssd` 是 Ascend 950 的定制设计方案，不是可在 Atlas A2/A3 上直接实例化的公共 Device adapter。

## 组件链

`include/attention/catlass_adapter.hpp` 在显式定义 `ATTENTION_ENABLE_CATLASS_DEVICE_ADAPTER` 后才包含 CATLASS/CANN 头文件。默认路径只依赖 C++ 标准库，可供 host reference 使用。

| 阶段 | 数学关系与逻辑形状 | A/B 的 CATLASS layout | 输出 |
|---|---|---|---|
| BC | `C^T[L,N] @ B[N,L]` | `ColumnMajor / RowMajor` | FP32 `M[L,L]` |
| StateC | `StateSnapshot[P,N] @ C[N,L]` | `RowMajor / RowMajor` | FP32 `YStateRaw[P,L]` |
| WX | `X[P,L] @ W^T[L,L]` | `RowMajor / ColumnMajor` | FP32 `YIntra[P,L]` |
| XB | `X[P,L] @ BTilde^T[L,N]` | `RowMajor / ColumnMajor` | FP32 `StateChunk[P,N]` |

四个别名均使用下列真实组件：

| 组件 | 选择 |
|---|---|
| ArchTag | `Catlass::Arch::AtlasA2` |
| DispatchPolicy | `Catlass::Gemm::MmadAtlasA2Pingpong<true>` |
| TileShape | `GemmShape<128,256,256>` / `GemmShape<128,256,64>` |
| 操作数 | FP16 A/B，FP32 C；`GemmType<Element, Layout>` |
| Block | `Gemm::Block::BlockMmad<...>` |
| Epilogue | `void`；衰减、因果、归一化和状态合并不属于通用 GEMM |
| Scheduler | `GemmIdentityBlockSwizzle<3,0>`，沿用 basic 示例，尚未按 shape 调优 |
| Kernel | `Gemm::Kernel::BasicMatmul<BlockMmad, void, BlockScheduler>` |
| Device 调用 | `BasicFp16Gemm<...>::Run()` 先经 `Kernel::ToUnderlyingArguments()` 补全布局，再直接执行 `Kernel{}(params)`，不使用 `DeviceGemm` |

`MakeChunkGemmArguments()` 将公开 `(P,N,L)` 和 GM 地址转换为四组真实 `Kernel::Arguments{GemmCoord, ptrA, ptrB, ptrC}`。实际调用方仍需分别执行 `CanImplement`、`GetWorkspaceSize`、`Initialize` 和 `operator()(stream, blockDim)`，并处理返回状态；本头文件不伪造 launch，也不选择物理卡。

## Host 数值契约与 device 数据流

`ChunkScanHostReferenceParams` 是独立的 host-only 全 FP32 输入/输出契约。它不继承 `BasicMatmul::Arguments`，也不宣称 CATLASS 能直接消费这些 host 指针。host reference 应按 `design.md` 直接计算完整因果衰减和状态递推，作为 device 的数值真值。

device 侧的依赖链为：

```text
C,B --BC--> M --Vector 因果/衰减/delta--> W --WX--> YIntra
FP32 StateIn --独立低精度快照--> StateSnapshot --StateC--> YStateRaw
YStateRaw --Vector exp(cumsum)--> YState
B,delta,cumsum --Vector--> BTilde --XB--> StateChunk
StateChunk + FP32 StateIn --Vector FP32 更新--> FP32 StateOut
YIntra + YState --Vector FP32 合并及输出转换--> Y
```

`StateSnapshot` 只是 StateC 的输入副本。FP16 快照和输出快照的舍入都不得回灌到 FP32 递推状态。`W` 和 `BTilde` 在进入后续 GEMM 前也需按该数据流生成低精度副本；host reference 始终保留 FP32 中间值。

## BC GEMM 真实 NPU 验证

`examples/catlass_chunk_gemm.cpp` 固定 `L=N=128`，以单 AIC 的 `<<<1, nullptr, stream>>>` 启动，kernel 内只调用 `BCGemm::Run`，不经过 `DeviceGemm`。Host 使用 ACL 分配并拷贝 FP16 B/C GM，回收 FP32 M 后按 `M[i,j]=sum_n C[n,i]*B[n,j]` 做 FP32 数值校验。

2026-11-27 在 Atlas A2/A3 目标的 device 0、CANN 9.2.0 环境实测通过。执行前 `npu-smi info -t proc-mem -i 0` 显示无进程。构建采用 `bisheng -x asc -std=c++17 --npu-arch=dav-2201 -DCATLASS_ARCH=2201`，加入项目 `include/`、CATLASS `include/` 和 CANN include，并使用官方 `00_basic_matmul` 的 `-mllvm -cce-aicore-record-overflow=true`、`-mllvm -cce-aicore-dcci-insert-for-scalar=false` 及以下链接库：

```text
ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager ascendcl
 tiling_api nnopbase platform dl ascendalog unified_dlog
```

实测输出：

```text
BC shape: M=128, N=128, K=128
max_absolute_error=0, mismatches=0
CATLASS BC GEMM PASS
```

该结果只覆盖固定对齐 shape 的 BC GEMM 数值正确性，不是性能结论。

## 未验证边界

- StateC、WX、XB 尚未分别做 device 数值验证；小 shape、非对齐尾块和其他 Tile/Swizzle 配置也未实测。
- `BasicMatmul` 是四次独立 GEMM，不提供 Mamba 设计中的片上 chunk 常驻状态、AIC/AIV 融合或跨 kernel 自动同步。调用方必须按上述依赖顺序同步，且不得并发覆盖中间缓冲区。
- host reference 目前只有参数契约，没有在此文件中实现数值函数；完整 CPU golden、非法指针/shape 检查及多 chunk 测试仍由公共原语层负责。
- BF16、动态 Tile/Swizzle 选择、融合 kernel、workspace 复用和性能结论均不在首版已验证范围内。
