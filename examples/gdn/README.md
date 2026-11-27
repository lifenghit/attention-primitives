# GDN example

演示标量门控 Gated Delta Rule 的两个 token 递推和最终状态输出。接口：`attention::linear::GdnParams` / `gatedDeltaRule`。

```sh
cmake -S . -B build
cmake --build build --target gdn_example
./build/examples/gdn_example
```
