# 四机开发与验证记录

## 环境与调度

- 四台远端服务器分别运行 vLLM 容器，每节点发现 16 张 Ascend910。
- 实际任务通过 AReaL-RSI `JobExecutor` 并发提交，独立构建目录和持久化 JobRecord。
- CANN 9.2.0，编译目标 `dav-2201`，SoC `Ascend910_9382`。
- CATLASS 参考提交 `3a6b66970ea1469fb566eb5f50afa9a469606e92`。

## 并行工作包

| 工作包 | 节点 | JobRecord | 结果 |
| --- | --- | --- | --- |
| 稀疏 C++ 严格编译和运行 | 33.215.119.129 | `attention-v3:sparse-cpp-verified` | succeeded，exit 0，资源释放 |
| 线性 C++ 严格编译和运行 | 33.215.119.117 | `attention-v1:linear-cpp` | succeeded，exit 0，资源释放 |
| Python 全套 + CMake/CTest | 33.215.118.0 | `attention-v1:reference-suite` | 6 项 Python、2 项 CTest 全通过 |
| CATLASS bisheng 编译 | 33.215.118.190 | `attention-v7:catlass-asc-language` | succeeded，exit 0，资源释放 |
| 四机回归 | 四节点 | `attention-regression:all-node-*` | 四份均 succeeded，exit 0，资源释放 |

稀疏测试初版期望值曾同时误解均值池化公式和 causal 索引，已按冻结契约修正后复跑通过；没有修改实现来迎合错误测试。

## 真实 NPU 结果

最小 BC GEMM 使用 `BCGemm::Run`，经 `Kernel::ToUnderlyingArguments()` 构造布局后直接 `Kernel{}(params)`，不使用 DeviceGemm。固定 `M=N=K=128`，FP16 输入、FP32 输出，主机逐元素校验 16384 个结果：

```text
BC shape: M=128, N=128, K=128
max_absolute_error=0, mismatches=0
worst_index=0, expected=0.0273438, actual=0.0273438
M[0]=0.0273438 M[1]=0.0456543 M[2]=-0.0522461 M[3]=-0.0629883
CATLASS BC GEMM PASS
```

二进制：`/storage/lf02381138/attention-primitives/build-catlass/catlass_chunk_gemm`。

## 未验收边界

- StateC、WX、XB 仅完成真实宏路径编译，尚未分别进行 device 数值验收。
- 非对齐 shape、尾块和 BF16 未进行 device 验收。
- indexer/top-k 当前是确定性 host/reference 实现，尚无独立 NPU kernel 性能结论。
- 四次独立 BasicMatmul 不是 Mamba SSD 的片上融合流水线；未测试跨 chunk 常驻状态、CrossCoreFlag 或性能。
- 当前没有生产性能结论，也没有完整 FlashAttention 验收结论。
