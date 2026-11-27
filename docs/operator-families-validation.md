# 六类 Attention 算子验证记录

## 实现与公共组件

- `kda.hpp/.py`：逐维门控 KDA、GVA、delta-rule 状态和最终状态。
- `gdn.hpp/.py`：标量门控 GDN。
- `gla.hpp/.py`：逐维门控 GLA，无 beta 路径。
- `dsa.hpp/.py`：weighted-ReLU token indexer、共享选择、GQA softmax。
- `msa.hpp/.py`：组独立 block max-pool、强制局部块、块内 causal softmax。
- `qsa.hpp/.py`：ReLU 多头 score、ratio 可见块、压缩 K/V、物理页映射。
- `sparse_common.py`：DSA/MSA/QSA 共用稳定 top-k 和局部块选择，统一同分索引升序及 -1 padding。

## 直接 FLA 差分

参考：FLA upstream commit `42c22207f6767e7cf5e46e89be70fd0d9040776d`，Python 2.10.0+cpu。

JobRecord `attention-families-v1:fla-alignment`：succeeded，exit 0，资源释放。

| 用例 | 对比内容 | 结果 |
| --- | --- | --- |
| KDA + GVA + 非零初态 | 输出与最终状态，atol/rtol=1e-5 | 通过 |
| GDN 标量门控 | 输出与最终状态 | 通过 |
| GLA 逐维门控 | 输出与最终状态 | 通过 |
| DSA weighted-ReLU + GQA | top-1 token 集合与主输出 | 通过 |

FLA `torch.topk` 对 ReLU 零分并列不保证稳定顺序。公共组件采用确定的索引升序；直接差分选择 top-1 避免无意义并列，未修改公共语义去匹配未定义顺序。

## C++ 与公开语义验证

- `attention-families-v2:linear-cpp`：KDA/GDN/GLA 严格 C++17 编译与运行全部通过。
- `attention-families-v2:sparse-cpp`：DSA/MSA 通过；QSA 初版测试数据错误已修正后单独复验。
- `attention-families-v4:qsa-cpp-verified`：QSA 压缩、ratio 边界、稳定选择、物理页错误、FP32 softmax 全部通过。
- `attention-families-v2:family-regression`：公共选择组件和原有六组件回归通过。

MSA 使用 FLA `ops/dsa/README.md` 的公开公式和 MiniMax MSA 语义验证。QSA 使用保存的 vendor 生产公式验证。它们当前属于公开语义对齐，不宣称与生产 fused kernel 做过了逐元素差分。

## 边界

本轮交付为 FP32 host/reference 算子和公共组件完善。未新增独立 NPU kernel，不宣称六算子的 NPU 精度或性能通过。CATLASS BC 真机结果沿用 `docs/validation.md`，不自动覆盖新算子。后续需要为六算子分别完成 device 实现、FLA-NPU/生产 kernel 同输入差分、非对齐 shape、varlen、BF16 和性能验收。
