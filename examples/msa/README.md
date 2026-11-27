# MSA example

演示 block max-pool、每 GQA 组 top-k 和强制局部块选择。接口：`attention::MSAParams` / `msa`。

```sh
cmake -S . -B build
cmake --build build --target msa_example
./build/examples/msa_example
```
