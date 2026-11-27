# KDA/GDN/GLA 与 DSA/MSA/QSA 接口契约 v1

## 参考来源

- FLA 上游 commit `42c22207f6767e7cf5e46e89be70fd0d9040776d`：`fla/ops/{kda,gated_delta_rule,gla,dsa}/naive.py`。
- FLA-NPU 仓内 `fla/ops/ascendc/{kda,gdn}` 的公开文档。
- QSA：`flagos-ai/vllm-plugin-FL` commit `fadbba0ea59bbaa46c77b06d465321ab88b44643` 的 vendor reference，以及 FlagGems 保存的 SHA256 `40a3190b04af28a1bc5944e29a334b041384ee6cc12bdd8c8a7382e23b207342`。
- MSA：FLA 上游 `fla/ops/dsa/README.md` 和 MiniMax 公开 MSA 语义，block max-pool、每 GQA 组 top-k、强制当前块。

首版提供 FP32 host/reference 算子和独立 C++ 头文件。NPU/CATLASS 复用 block_score/topk/block_selection/chunk/state 组件，性能声明另行验收。

## 线性公共布局与状态

公开张量采用 sequence-major `[B,T,H,D]`，单 batch/head 的内部视图采用 `[T,D]`。状态统一 `[K,V]` FP32，输出 `[T,V]`，支持可选非零初态和最终状态。query scale 默认 `K^-0.5`。

### KDA

- q/k heads=H，v heads=HV，HV 是 H 的整数倍，q/k 按 G=HV/H repeat。
- g `[T,HV,K]` 为逐维 log gate，beta `[T,HV]`。
- 每步先 `S=S*exp(g)`，再 `delta=v-(k@S)`，然后 `S+=beta*k⊗delta`，输出 `q@S`。

### GDN

- q/k/v heads 相同，g/beta `[T,H]` 为标量 log gate 和 delta 系数。
- 与 KDA 相同的 delta-rule 更新，但 gate 对整个 `[K,V]` 状态标量广播。

### GLA

- gk `[T,H,K]` 为逐维 log gate，无 beta。
- `S=S*exp(gk)+k⊗v`，输出 `q@S`。

## 稀疏公共语义

主 Attention 使用 FP32 score、稳定 softmax 和 causal mask。选中集合之外的位置必须为 -inf。GQA 按 KV head 映射，indexer 的选择是否跨 query head 共享按各算子定义。

### DSA

- q_idx `[T,HI,DI]`，k_idx `[T,DI]`，w_idx `[T,HI]` 可选。
- `I[t,s]=sum_h w[t,h]*ReLU(scale*q_idx[t,h]@k_idx[s])`；无 w 时等权求和。
- 每个 query 在 causal token 集合中选 top-k，所有主 query heads 共享选择。
- top-k 排序与公共 stable top-k 一致，不可见位置填 -1；主分支使用完整 q/k/v GQA softmax。

### MSA

- 每个 GQA 组独立 indexer，token score 无 ReLU，按 block_size max-pool。
- 每组选 topk_blocks，强制当前 query block 入选；主分支只在选中块的 causal token 上精确 softmax。
- block_size/topk 可配置，128/16 为生产配置而非测试硬编码。

### QSA

- index q 多 head、共享 index k；score 为各 head ReLU(dot) 求和后除 score_divisor。
- 每 compress_ratio 个 token 形成一个 index block；query 可见块数为 `min((position+1)/ratio, sequence_length/ratio)` 的整数下界。
- 支持 group mean-pooling 形成压缩 K/V，paged block table 按逻辑块映射。
- 首版 selection 复用稳定 top-k；主分支只读选中块，非法物理页报错。

## 验收

- 与上述参考 naive/recurrent 在相同 FP32 输入上比较输出和状态。
- KDA 额外覆盖 GVA；GDN 覆盖标量门控；GLA 覆盖无 beta 路径。
- DSA 验证共享 token 选择；MSA 验证组独立 max-pool 和强制局部块；QSA 验证 ratio 可见块、压缩和分页。
- 浮点输出使用 FP32 容差 `atol=rtol=1e-5`；索引完全一致。
- NPU 只声明实际编译/运行过的路径，host 对齐不自动等价于 NPU 对齐。
