#define ATTENTION_ENABLE_CATLASS_DEVICE_ADAPTER
#include "attention/catlass_adapter.hpp"

int main()
{
    const auto shapes = attention::MakeChunkGemmShapes(64, 128, 128);
    return shapes.bc.m == 128 && shapes.stateC.k == 128 ? 0 : 1;
}
