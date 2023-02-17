# ByteTransformer: optimized BERT transformer inference on NVIDIA GPU

## Supported Models

Currently only standard BERT transformer encoder is available under this repo.

## Environment requirements
* CUDA: 11.6
* CMake >= 3.13
* PyTorch >= 1.8
* GPU compute capability: 7.0(V100) / 7.5(T4) or 8.0(A100)
* Python >= 3.7

Tested on: A100 + CUDA 11.6 + PyTorch 1.13.0+cu116 + Python 3.9.16

## Features
* Support remove padding
* Hand-written fused attention (for seqlen <= 128, with WMMA api)
* CUTLASS fused attention (for seqlen > 128, only available for GPU Arch 8.0)
* Support Pytorch op extension

## Build
```bash
mkdir build && cd build
cmake -DDataType=FP16 -DBUILD_THS=ON -DCUDAARCHS="80" ..
make
```

## Unit Test
### cpp
Run the following code to generate test data
```bash
cd build
# bs seqlen heads head_size
python3 bert_transformer_test.py 16 64 12 64 --avg_seqlen 32 --dtype fp16 --export_data
```

After test data generated (*.in, *.out files under current directory), then run
```
./bin/bert_transformer_test 16 64 12 64
```

### Pytorch
Use the same script, but without `--export_data` flag.
```bash
# bs seqlen heads head_size
python3 bert_transformer_test.py 16 64 12 64 --avg_seqlen 32 --dtype fp16
```

## Benchmark
```bash
cd build
../benchmark/bert_bench.sh
```
