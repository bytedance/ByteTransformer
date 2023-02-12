// Copyright 2023 Bytedance Ltd. and/or its affiliates.
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "bytetransformer/include/gemm_bias_act.h"

namespace bytetransformer {
template <>
__global__ void add_bias_gelu<float>(float *output, const float *bias, const int M, const int N) {
  int row_offset = blockIdx.x * N;
  for (int tid = threadIdx.x; tid < N; tid += blockDim.x) {
    float out = output[row_offset + tid] + __ldg(&bias[tid]);
    output[row_offset + tid] = gelu(out);
  }
}

template <>
__global__ void add_bias_gelu<__half>(__half *output, const __half *bias, const int M,
                                      const int N) {
  if (N % 4 != 0) {
    half2 *output_ptr = (half2 *)output;
    const half2 *bias_ptr = (const half2 *)bias;

    int row_offset = blockIdx.x * N / 2;
    for (int tid = threadIdx.x; tid < N / 2; tid += blockDim.x) {
      half2 out = __hadd2(output_ptr[row_offset + tid], __ldg(&bias_ptr[tid]));
      output_ptr[row_offset + tid] = gelu(out);
    }
  } else {
    float2 *output_ptr = (float2 *)output;
    const float2 *bias_ptr = (const float2 *)(bias);
    int row_offset = blockIdx.x * N / 4;
    for (int tid = threadIdx.x; tid < N / 4; tid += blockDim.x) {
      half4 val, bias_val;
      val.x = output_ptr[row_offset + tid];
      bias_val.x = __ldg(&bias_ptr[tid]);

      val.h[0] = gelu(__hadd2(val.h[0], bias_val.h[0]));
      val.h[1] = gelu(__hadd2(val.h[1], bias_val.h[1]));

      output_ptr[row_offset + tid] = val.x;
    }
  }
}

void cublas_gemm_bias_gelu(const __half *A, const __half *B, __half *C, const __half *bias, int m,
                           int k, int n, cudaStream_t stream, cublasHandle_t cublas_handle,
                           int cublasAlgo) {
  dense_layer_kernel_launcher(A, B, C, m, k, n, cublas_handle, stream, cublasAlgo);
  add_bias_gelu<<<m, n / 8, 0, stream>>>(C, bias, m, n);
}

template <>
void gemm_bias_gelu<float>(const float *A_, const float *B_, float *C_, const float *bias_, int m_,
                           int k_, int n_, cudaStream_t stream, cublasHandle_t cublas_handle,
                           int cublasAlgo, int arch) {
  dense_layer_kernel_launcher(A_, B_, C_, m_, k_, n_, cublas_handle, stream, cublasAlgo);
  add_bias_gelu<<<m_, n_ / 4, 0, stream>>>(C_, bias_, m_, n_);
}

template <>
void gemm_bias_gelu<__half>(const half *A_, const half *B_, half *C_, const half *bias_, int m_,
                            int k_, int n_, cudaStream_t stream, cublasHandle_t cublas_handle,
                            int cublasAlgo, int arch) {
  if (m_ < 8)
    cublas_gemm_bias_gelu(A_, B_, C_, bias_, m_, k_, n_, stream, cublas_handle, cublasAlgo);
  else {
#if (__CUDACC_VER_MAJOR__ >= 11)
    const ElementComputeEpilogue alpha = ElementComputeEpilogue(1.0f);
    const ElementComputeEpilogue beta = ElementComputeEpilogue(1.0f);
    const int split_k_slices_ = 1;
    void *cutlass_workspace_ = nullptr;
    cutlass::gemm::GemmCoord problem_size(m_, n_, k_);

    if (arch == 70 && (m_ > 896 && m_ <= 7936)) {
      using SmArch = cutlass::arch::Sm70;
#define _inst_m 8
#define _inst_n 8
#define _inst_k 4
      constexpr int NumStages = 2;
      GEMM_TYPE(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n, _inst_k,
                NumStages)
      GEMM_BIAS_GELU(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n,
                     _inst_k)
      gemmBiasGelu_op;
      GEMM_INIT(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n, _inst_k)
#undef _inst_m
#undef _inst_n
#undef _inst_k
      CUTLASS_CHECK(gemmBiasGelu_op.initialize(args, cutlass_workspace_));
      CUTLASS_CHECK(gemmBiasGelu_op(stream));
    } else if (arch == 75 && (m_ > 192 && m_ <= 3456)) {
      using SmArch = cutlass::arch::Sm75;
#define _inst_m 16
#define _inst_n 8
#define _inst_k 8
      constexpr int NumStages = 2;
      GEMM_TYPE(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n, _inst_k,
                NumStages)
      GEMM_BIAS_GELU(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n,
                     _inst_k)
      gemmBiasGelu_op;
      GEMM_INIT(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n, _inst_k)
#undef _inst_m
#undef _inst_n
#undef _inst_k
      CUTLASS_CHECK(gemmBiasGelu_op.initialize(args, cutlass_workspace_));
      CUTLASS_CHECK(gemmBiasGelu_op(stream));
    } else if (arch == 80 && (m_ >= 384 && m_ <= 16384))  // < 19742
    {
      using SmArch = cutlass::arch::Sm80;
#define _inst_m 16
#define _inst_n 8
#define _inst_k 16
      constexpr int NumStages = 3;
      GEMM_TYPE(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n, _inst_k,
                NumStages)
      GEMM_BIAS_GELU(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n,
                     _inst_k)
      gemmBiasGelu_op;
      GEMM_INIT(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n, _inst_k)
#undef _inst_m
#undef _inst_n
#undef _inst_k
      CUTLASS_CHECK(gemmBiasGelu_op.initialize(args, cutlass_workspace_));
      CUTLASS_CHECK(gemmBiasGelu_op(stream));
    } else
      cublas_gemm_bias_gelu(A_, B_, C_, bias_, m_, k_, n_, stream, cublas_handle, cublasAlgo);
#else
    cublas_gemm_bias_gelu(A_, B_, C_, bias_, m_, k_, n_, stream, cublas_handle, cublasAlgo);
#endif
  }
}

template <>
void gemm_bias_relu<float>(const float *A_, const float *B_, float *C_, const float *bias_, int m_,
                           int k_, int n_, cudaStream_t stream, cublasHandle_t cublas_handle,
                           int cublasAlgo, int arch) {
  dense_layer_kernel_launcher(A_, B_, C_, m_, k_, n_, cublas_handle, stream, cublasAlgo);
  // add_bias_relu<<<m_, n_ / 4, 0, stream>>>(C_, bias_, m_, n_);
}

template <>
void gemm_bias_relu<__half>(const half *A_, const half *B_, half *C_, const half *bias_, int m_,
                            int k_, int n_, cudaStream_t stream, cublasHandle_t cublas_handle,
                            int cublasAlgo, int arch) {
}
}  // namespace bytetransformer
