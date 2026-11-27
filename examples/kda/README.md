# KDA example

演示逐维门控 KDA 的两个 token 递推，包含 delta-rule 状态更新和最终状态输出。接口：`attention::linear::KDAParams` / `kdaRecurrence`。

```sh
cmake -S . -B build
cmake --build build --target kda_example
./build/examples/kda_example
```
