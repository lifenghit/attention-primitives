# GLA example

演示逐维 `exp(gk)` 衰减的 GLA 状态更新，无 beta 分支。接口：`attention::linear::GLAParams` / `glaRecurrence`。

```sh
cmake -S . -B build
cmake --build build --target gla_example
./build/examples/gla_example
```
