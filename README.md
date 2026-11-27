# Attention 公共原语

本目录提供稀疏与线性 Attention 公共原语及 CATLASS 配合层。接口语义以 `docs/design.md` 为准。

- `attention/block_score.hpp`：FP16 block 均值池化、GQA、causal、FP32 score。
- `attention/topk.hpp`：确定性 top-k，同分按索引升序，支持空行和 padding。
- `attention/block_selection.hpp`：逻辑块到物理 block table 的安全映射。
- `attention/chunk_scan.hpp`：chunk 内衰减、历史状态读出和 FP32 输出。
- `attention/state_update.hpp`：非零初态、跨 chunk FP32 状态递推。
- `attention/catlass_adapter.hpp`：BC、StateC、WX、XB 四次 GEMM 的 CATLASS 组件和参数封装。
- `attention/{kda,gdn,gla}.hpp`：三类线性 Attention FP32 递推与状态。
- `attention/{dsa,msa,qsa}.hpp`：三类稀疏 Attention 的 indexer、选择与主分支。
- `python/attention_primitives/sparse_common.py`：共享稳定 top-k 和局部块选择。

六类算子的接口见 `docs/operator-families-api.md`，FLA 差分和 C++ 结果见 `docs/operator-families-validation.md`。

## Host CPU 构建与测试

依赖仅有 CMake、支持 C++17 的 C++ 编译器和 Python 3。C++ 库不依赖第三方库；CPU 测试只使用 Python 标准库。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

也可以单独运行测试：

```sh
python3 tests/run_cpu_tests.py sparse
python3 tests/run_cpu_tests.py linear
python3 tests/run_cpu_tests.py all
```

CTest 固定提供 `test_sparse_cpu` 和 `test_linear_cpu`。这些测试验证设计公式、确定性 top-k、块映射、chunk/state 递推及边界条件；它们是独立的 CPU 参考契约，不代替对后续 C++ 实现或 device 输出的差分验收。

`attention_primitives` 是 `INTERFACE` 库，别名是 `attention_primitives::attention_primitives`。它向使用方导出 `include/` 和 `cxx_std_17`，不假定其中已有某个具体头文件或 API。

## 六类算子示例

`examples/kda`、`gdn`、`gla`、`dsa`、`msa`、`qsa` 分别提供独立 `main.cpp` 和说明。默认随 CMake 构建，可用 `-DATTENTION_BUILD_EXAMPLES=OFF` 关闭：

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

也可以单独执行 `cmake --build build --target kda_example`，然后运行 `./build/examples/kda_example`。示例仅演示公开接口，完整差分和边界测试见 `tests/`。

## 单次远程作业

`scripts/remote_job.py` 通过本机 `ssh` 客户端在一个主机上执行一条命令，并将 stdout、stderr 和退出码封装为 JSON。Python 代码只使用标准库；SSH 认证由本机现有配置提供，非交互执行。

```sh
python3 scripts/remote_job.py \
  --host user@server.example \
  --command 'cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure'
```

可用 `--port`、`--connect-timeout`、`--timeout` 控制连接与执行时限，用 `--dry-run` 仅打印调用计划。该脚本不是调度器：没有队列、重试、主机选择、NPU 分配或共享构建目录管理。并发工作包必须使用各自独立的远程构建目录。

## 目录

- `include/`：公共 C++ 头文件，由原语实现提供。
- `docs/design.md`：布局、公式和验收语义的唯一契约。
- `docs/catlass-integration.md`：CATLASS 组件链、编译参数和边界。
- `docs/validation.md`：四机 JobRecord、CTest 和真实 NPU 验证结果。
- `tests/run_cpu_tests.py`：无第三方依赖的稀疏/线性 CPU 参考测试。
- `scripts/remote_job.py`：单主机、单命令 SSH JSON 包装器。
- `CMakeLists.txt`：host-only C++17 接口目标和两个 CTest 入口。

## CATLASS / NPU 边界

本 CMake 工程只配置 host CPU，不查找、下载或链接 CATLASS/CANN，不生成 NPU kernel，也不访问物理卡。后续 CATLASS GEMM/Tile 适配应放在独立的 NPU 构建入口中，由明确的 CANN/CATLASS 工具链和受控资源执行；不得把 host 构建成功表述为 NPU 编译、精度或性能通过。FP16/BF16 device 数值验收及性能结论必须来自目标设备实测。
