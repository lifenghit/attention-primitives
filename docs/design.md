# Attention 公共原语设计契约 v1

## 目标与边界

提供可组合的 Attention 原语，不在首版重复实现完整 FlashAttention。稀疏路径由 block score/indexer、稳定 top-k、块选择组成；线性路径由 chunk scan、chunk 输出和状态更新组成。CATLASS 负责 GEMM/Tile 数据搬运，公共原语负责注意力特有索引、因果约束、衰减和状态语义。

首版目标为 Atlas A2/A3（`CATLASS_ARCH=2201`），FP16 输入、FP32 索引分数与递推状态。BF16 在接口中预留，不作为首轮验收前提。

## 稀疏原语

### BlockScoreParams

- Q：`[QBlocks, QHeads, D]`，行主序 FP16。
- K：`[KVBlocks, KVHeads, D]`，行主序 FP16。
- QHeads 必须是 KVHeads 的整数倍。
- 输出 scores：`[QBlocks, QHeads, KVBlocks]`，FP32。
- 每个 Q/K block 的 D 维向量先做 FP32 均值池化，再计算缩放点积 `dot/sqrt(D)`。
- `causal=true` 且 `kvBlock>qBlock` 时输出 `-INFINITY`。

### TopKParams

- 输入 scores 同上。
- 每行选择最多 `min(topK, KVBlocks)` 个索引，输出 `[QBlocks, QHeads, topK]` int32 indices 和 counts。
- 按分数降序；分数相同按 KV block 索引升序；`-INFINITY` 不入选。
- `topK>0`，空行 count=0，未使用槽位写 -1。
- 首版采用确定性单核参考实现，之后优化并行堆/归并，语义不得改变。

### BlockSelectionParams

- 输入 top-k 逻辑索引和可选物理 block table。
- 输出物理块号、有效计数和用于 gather 的密集位置。
- 无 block table 时恒等映射；越界物理块号为非法输入并返回错误，禁止静默读越界。

## 线性注意力原语

固定一个 batch/head，chunk 长度 L、通道 P、状态维度 N。

### ChunkScanParams

- X `[P,L]`、B `[N,L]`、C `[N,L]`、delta `[L]`。
- `cumsum[t]=sum_{j<=t} A*delta[j]`，A 为标量衰减参数。
- 输出历史状态读出与 chunk 内输出之和，使用 FP32 累加。
- `M=C^T@B`；`W[i,j]=M[i,j]*exp(cumsum[i]-cumsum[j])*delta[j]`（j<=i，否则 0）。
- `YIntra=X@W^T`；`YState[t]=exp(cumsum[t])*(StateIn@C)[t]`。

### StateUpdateParams

- StateIn/StateOut `[P,N]`，外部快照 FP16，内部递推 FP32。
- `BTilde[:,j]=B[:,j]*delta[j]*exp(cumsum[L-1]-cumsum[j])`。
- `StateOut=exp(cumsum[L-1])*StateIn+X@BTilde^T`。
- 支持零/非零初态、L 非整 chunk 尾段、多个 chunk 顺序调用。
- 状态快照舍入不得反馈到内部 FP32 递推。

## CATLASS 配合

- GEMM 使用 CATLASS Kernel/Block/Tile，不在原语适配层另写通用矩阵乘框架。
- 参考：`examples/40_flash_attention_infer_tla`（paged 索引和 FA）、`examples/78_x_attention`（共享/独立 KV 合并）、`examples/108_ascend950_mamba_ssd`（chunk/state 语义）。
- 首版 indexer/top-k 可使用 Vector 原语；chunk/state 的 GEMM 适配使用 CATLASS 组件。
- Host 负责布局、非法 shape 和非法索引检查；device 不处理公开布局转换。

## 验收

1. NumPy/PyTorch CPU golden 与 device 输出对比。
2. 稀疏：空行、topK=1/topK>K、同分、causal、GQA、物理块映射和非法索引。
3. 线性：L=1/非整块、多个 chunk、零/非零初态、衰减极值、状态连续性。
4. 四台服务器分别运行独立构建/测试工作包；共享只读源码，不写共享构建目录。
5. FP16 输出采用混合容差，索引必须完全一致；记录原始日志和环境指纹。性能结论只来自真实 device 测量。
