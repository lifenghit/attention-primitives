# QSA example

演示 compress_ratio 分组、ReLU 多头 score、稳定选择和物理块映射。接口：`attention::QSAParams` / `qsa`。

```sh
cmake -S . -B build
cmake --build build --target qsa_example
./build/examples/qsa_example
```
