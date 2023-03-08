# ByteTransformer: Optimized BERT Transformer Inference on NVIDIA GPUs

## Introduction

ByteTransformer is a high-performance inference library for BERT-like transformers that offers the following features:

* Provides Python and C++ APIs, with the PyTorch plugin allowing users to enhance transformer inference with just a few lines of Python code.
* Supports both fixed-length and variable-length transformers.
* Includes end-to-end architectural-aware optimizations for the padding-free algorithm on BERT routines, including QKV encoding, softmax, feed forward network, activation, layernorm, and multi-head attention.

ByteTransformer has been widely deployed to improve in-house transformer inference serving systems at ByteDance, delivering superior performance over other transformer implementations for both fixed-length and variable-length inputs. The technical details have been published at IEEE IPDPS 2023.

## Cite Us

If you use our library, please cite our research paper.

```
@article{zhai2022bytetransformer,
  title={ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs},
  author={Zhai, Yujia and Jiang, Chengquan and Wang, Leyuan and Jia, Xiaoying and Zhang, Shang and Chen, Zizhong and Liu, Xin and Zhu, Yibo},
  journal={arXiv preprint arXiv:2210.03052},
  year={2022}
}
```

## Performance and Speedup

We compared ByteTransformer with PyTorch, TensorFlow, FasterTransformer, and DeepSpeed on an A100 GPU. The benchmark script is available in [benchmark/bert_bench.sh](https://github.com/bytedance/ByteTransformer/blob/main/benchmark/bert_bench.sh).

**1. Standard BERT batch size = 1, average sequence length = 0.6 * maximal, execution time in millisecond:**

|      | PyTorch | Tensorflow | FasterTransformer | FasterTransformer with remove padding | DeepSpeed | ByteTransformer |
|------|-------------|----------------|-------------------|---------------------------------------|---------------------|-----------------|
| 64   | 2.93        | 2.46           | 1.05              | 1.23                                  | 1.17                | 0.90            |
| 128  | 3.18        | 2.6            | 1.10              | 1.43                                  | 1.28                | 0.97            |
| 192  | 3.18        | 2.81           | 1.26              | 1.43                                  | 1.40                | 1.36            |
| 256  | 2.81        | 2.9            | 1.35              | 1.55                                  | 1.51                | 1.43            |
| 320  | 3.11        | 3.24           | 1.63              | 1.66                                  | 1.84                | 1.69            |
| 384  | 2.87        | 3.43           | 1.64              | 1.64                                  | 1.95                | 1.72            |
| 448  | 2.99        | 3.61           | 2.26              | 2.35                                  | 2.23                | 1.86            |
| 512  | 2.89        | 3.74           | 2.28              | 2.43                                  | 2.37                | 2.00            |
| 576  | 2.99        | 4.03           | 2.51              | 2.59                                  | 2.70                | 2.19            |
| 640  | 2.99        | 4.54           | 2.85              | 2.83                                  | 3.17                | 2.23            |
| 704  | 3.21        | 4.67           | 3.16              | 3.44                                  | 3.32                | 2.47            |
| 768  | 3.33        | 4.88           | 3.26              | 3.63                                  | 3.46                | 2.51            |
| 832  | 3.78        | 5.39           | 3.75              | 3.87                                  | 3.97                | 2.80            |
| 896  | 3.86        | 5.81           | 4.08              | 4.95                                  | 4.37                | 2.86            |
| 960  | 4.02        | 6.27           | 4.30              | 5.23                                  | 4.66                | 3.12            |
| 1024 | 4.2         | 6.37           | 4.51              | 4.96                                  | 4.86                | 3.16            |


**2. Standard BERT batch size = 16, average sequence length = 0.6 * maximal, execution time in millisecond:**

|      | PyTorch | Tensorflow | FasterTransformer | FasterTransformer with remove padding | DeepSpeed | ByteTransformer |
|------|-------------|----------------|-------------------|---------------------------------------|---------------------|-----------------|
| 64   | 3.2         | 4.57           | 2.24              | 1.93                                  | 2.81                | 2.09            |
| 128  | 4.97        | 6.97           | 3.62              | 3.33                                  | 4.54                | 3.18            |
| 192  | 7.65        | 9.37           | 5.26              | 5.29                                  | 6.68                | 5.08            |
| 256  | 9.56        | 12.17          | 6.77              | 5.49                                  | 9.03                | 6.85            |
| 320  | 13.21       | 15.87          | 8.85              | 6.47                                  | 12.81               | 7.49            |
| 384  | 15.01       | 18.56          | 10.37             | 7.05                                  | 15.19               | 8.44            |
| 448  | 19.06       | 23.01          | 15.97             | 12.54                                 | 18.83               | 8.89            |
| 512  | 21          | 26.03          | 18.03             | 13.79                                 | 21.55               | 9.22            |
| 576  | 24.33       | 31.24          | 21.11             | 17.65                                 | 26.2                | 10.15           |
| 640  | 28.03       | 35.07          | 24.52             | 20.34                                 | 30.24               | 12.04           |
| 704  | 32.33       | 41.43          | 28.94             | 24.52                                 | 34.65               | 13.55           |
| 768  | 35.31       | 44.62          | 32.09             | 28.21                                 | 37.95               | 16.3            |
| 832  | 40.75       | 51.87          | 36.33             | 31.69                                 | 45.32               | 16.92           |
| 896  | 44.47       | 55.65          | 42.17             | 38.05                                 | 49.48               | 20.67           |
| 960  | 49.72       | 63.59          | 47.01             | 42.98                                 | 55.72               | 23.27           |
| 1024 | 53.21       | 65.94          | 50.28             | 45.22                                 | 59.96               | 24.70           |


## Supported Models

Currently, only the standard BERT transformer encoder is available under this repository.

## Environment requirements
* CUDA: 11.6
* CMake: >= 3.13
* PyTorch: >= 1.8
* GPU compute capability: 7.0(V100) / 7.5(T4) or 8.0(A100)
* Python: >= 3.7

Tested on: A100 + CUDA 11.6 + PyTorch 1.13.0+cu116 + Python 3.9.16

## Building from Source
To build from source, run the following commands:
```bash
mkdir build && cd build
cmake -DCUDA_PATH=/usr/local/cuda -DDataType=FP16 -DBUILD_THS=ON -DCUDAARCHS="80" ..
make
```

## Getting Started with Unit Tests
### Unit Tests in C++
To generate test data, run the following code:
```bash
cd build
# batch sz = 16, seqlen = 64, head num = 12, head sz = 64, avg seqlen = 32
python3 bert_transformer_test.py 16 64 12 64 --avg_seqlen 32 --dtype fp16 --export_data
```

Here, `16`, `64`, `12`, and `64` represent batch size, sequence length, number of heads, and head size, respectively. The `--avg_seqlen 32` flag is used to set the average sequence length, `--dtype fp16` sets the data type, and `--export_data` exports the test data.


After test data is generated (`*.in` and `*.out` files are saved under the current directory), run the following command:

```
./bin/bert_transformer_test 16 64 12 64
```

Here, the arguments represent the same parameters as used in generating the test data.

### Unit Tests in a PyTorch Plugin in Python

To perform the unit tests in a PyTorch plugin in Python, use the same script as for C++, but without the `--export_data` flag. Run the following command in the terminal:

```bash
# batch sz = 16, seqlen = 64, head num = 12, head sz = 64, avg seqlen = 32
python3 bert_transformer_test.py 16 64 12 64 --avg_seqlen 32 --dtype fp16
```

Again, the arguments represent the same parameters as used in generating the test data.

## Benchmark
```bash
cd build
../benchmark/bert_bench.sh
```