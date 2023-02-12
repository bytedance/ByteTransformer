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
#pragma once
#include "common.h"
#include "reduce.h"

namespace bytetransformer {
__inline__ __device__ void layernorm(float local_out, const void *gamma, const void *beta,
                                     float *out_ptr, int n, float *s_) {
  float sum = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0)
    s_[0] = __fdividef(sum, n);
  __syncthreads();

  local_out -= s_[0];
  float variance = blockReduceSum<float>(local_out * local_out);
  if (threadIdx.x == 0)
    s_[1] = rsqrtf(__fdividef(variance, n) + 1e-6f);
  __syncthreads();

  *out_ptr = local_out * s_[1] * __ldg(&((float *)gamma)[threadIdx.x]) +
             __ldg(&((float *)beta)[threadIdx.x]);
}

__inline__ __device__ void layernorm(half2 local_out, const void *gamma, const void *beta,
                                     half2 *out_ptr, int n, float *s_, bool use_fp32) {
  float2 local_out_fp2 = __half22float2(local_out);
  float t_sum = local_out_fp2.x + local_out_fp2.y;
  float sum = blockReduceSum<float>(t_sum);
  if (threadIdx.x == 0)
    s_[0] = __fdividef(sum, n);
  __syncthreads();

  local_out_fp2.x -= s_[0];
  local_out_fp2.y -= s_[0];
  float variance = 0.0f;
  if (threadIdx.x < n / 2)
    variance = local_out_fp2.x * local_out_fp2.x + local_out_fp2.y * local_out_fp2.y;
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0)
    s_[1] = rsqrtf(__fdividef(variance, n) + 1e-6f);
  __syncthreads();

  if (threadIdx.x < n / 2) {
    float2 gamma_val, beta_val;
    if (use_fp32) {
      gamma_val = __ldg(&((const float2 *)gamma)[threadIdx.x]);
      beta_val = __ldg(&((const float2 *)beta)[threadIdx.x]);
    } else {
      gamma_val = __half22float2(__ldg(&((const half2 *)gamma)[threadIdx.x]));
      beta_val = __half22float2(__ldg(&((const half2 *)beta)[threadIdx.x]));
    }

    local_out_fp2.x = local_out_fp2.x * s_[1] * gamma_val.x + beta_val.x;
    local_out_fp2.y = local_out_fp2.y * s_[1] * gamma_val.y + beta_val.y;
    *out_ptr = __float22half2_rn(local_out_fp2);
  }
}

template <typename T>
__global__ void input_layernorm(T *out, const T *input, const void *gamma, const void *beta, int n,
                                bool use_fp32);

template <typename T>
__global__ void input_compress_layernorm(T *out, const T *input, const void *gamma,
                                         const void *beta, int n, bool use_fp32, T *out2,
                                         const int *batch_idx, const int *word_idx);

template <const int ite>
__inline__ __device__ void layernorm_v2(float *local_out, float sum, const void *gamma,
                                        const void *beta, float *out_ptr, int n, float *s_) {
  float mean = blockReduceSum<float>(sum);
  if (threadIdx.x == 0)
    s_[0] = __fdividef(mean, n);
  __syncthreads();

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out[i] -= s_[0];
    var += local_out[i] * local_out[i];
  }

  float variance = blockReduceSum<float>(var);
  if (threadIdx.x == 0)
    s_[1] = rsqrtf(__fdividef(variance, n) + 1e-6f);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    out_ptr[col_id] =
        local_out[i] * s_[1] * __ldg(&((float *)gamma)[col_id]) + __ldg(&((float *)beta)[col_id]);
  }
}

template <const int ite>
__inline__ __device__ void layernorm_v2(float2 *local_out_fp2, float sum, const void *gamma,
                                        const void *beta, half2 *out_ptr, int n, float *s_,
                                        bool use_fp32) {
  float mean = blockReduceSum<float>(sum);
  if (threadIdx.x == 0)
    s_[0] = __fdividef(mean, n);
  __syncthreads();

  float variance = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_fp2[i].x -= s_[0];
    local_out_fp2[i].y -= s_[0];
    variance += local_out_fp2[i].x * local_out_fp2[i].x + local_out_fp2[i].y * local_out_fp2[i].y;
  }

  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0)
    s_[1] = rsqrtf(__fdividef(variance, n) + 1e-6f);
  __syncthreads();

  float2 gamma_val[ite], beta_val[ite];
  if (use_fp32) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + threadIdx.x;
      gamma_val[i] = __ldg(&((const float2 *)gamma)[col_id]);
      beta_val[i] = __ldg(&((const float2 *)beta)[col_id]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + threadIdx.x;
      gamma_val[i] = __half22float2(__ldg(&((const half2 *)gamma)[col_id]));
      beta_val[i] = __half22float2(__ldg(&((const half2 *)beta)[col_id]));
    }
  }

#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_fp2[i].x = local_out_fp2[i].x * s_[1] * gamma_val[i].x + beta_val[i].x;
    local_out_fp2[i].y = local_out_fp2[i].y * s_[1] * gamma_val[i].y + beta_val[i].y;
    out_ptr[i * blockDim.x + threadIdx.x] = __float22half2_rn(local_out_fp2[i]);
  }
}

template <const int ite>
__global__ void input_layernorm_v2(float *out, const float *input, const void *gamma,
                                   const void *beta, int n, bool use_fp32) {
  int offset = blockIdx.x * n;
  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int id = offset + (i * blockDim.x + threadIdx.x);
    local_out[i] = __ldg(&input[id]);
    sum += local_out[i];
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out, sum, gamma, beta, out + offset, n, s_);
}

template <const int ite>
__global__ void input_layernorm_v2(__half *out, const __half *input, const void *gamma,
                                   const void *beta, int n, bool use_fp32) {
  const half2 *input_ptr = (const half2 *)input;
  int offset = blockIdx.x * n / 2;

  float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int id = offset + (i * blockDim.x + threadIdx.x);
    local_out_fp2[i] = __half22float2(__ldg(&input_ptr[id]));
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((half2 *)out) + offset, n, s_, use_fp32);
}

template <const int ite>
__global__ void input_compress_layernorm_v2(float *out, const float *input, const void *gamma,
                                            const void *beta, int n, bool use_fp32, float *out2,
                                            const int *batch_idx, const int *word_idx) {
  int from_offset = __ldg(&word_idx[blockIdx.x]) * n;
  int offset = blockIdx.x * n;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = from_offset + col_id;
    local_out[i] = __ldg(&input[id]);
    out[offset + col_id] = local_out[i];
    sum += local_out[i];
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out, sum, gamma, beta, out2 + offset, n, s_);
}

template <const int ite>
__global__ void input_compress_layernorm_v2(__half *out, const __half *input, const void *gamma,
                                            const void *beta, int n, bool use_fp32, __half *out2,
                                            const int *batch_idx, const int *word_idx) {
  const half2 *input_ptr = (const half2 *)input;
  int from_offset = __ldg(&word_idx[blockIdx.x]) * n / 2;
  int offset = blockIdx.x * n / 2;

  float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    half2 temp = __ldg(&input_ptr[from_offset + col_id]);
    ((half2 *)out)[offset + col_id] = temp;
    local_out_fp2[i] = __half22float2(temp);
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((half2 *)out2) + offset, n, s_, use_fp32);
}

template <typename T>
__global__ void add_bias_input_layernorm(T *out, const T *input, const T *bias, const void *gamma,
                                         const void *beta, int n, bool use_fp32);

template <typename T>
__global__ void add_bias_input_layernorm_restore_output(const T *out, const T *input,
                                                        const T *bias, const void *gamma,
                                                        const void *beta, int n, bool use_fp32,
                                                        T *out2, const int *batch_idx,
                                                        const int *word_idx, const int seq_len);

template <const int ite>
__global__ void add_bias_input_layernorm_v2(float *out, const float *input, const float *bias,
                                            const void *gamma, const void *beta, int n,
                                            bool use_fp32) {
  int offset = blockIdx.x * n;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = offset + col_id;
    local_out[i] = (float)(out[id] + __ldg(&input[id]) + __ldg(&bias[col_id]));
    sum += local_out[i];
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out, sum, gamma, beta, out + offset, n, s_);
}

template <const int ite>
__global__ void add_bias_input_layernorm_v2(__half *out, const __half *input, const __half *bias,
                                            const void *gamma, const void *beta, int n,
                                            bool use_fp32) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  int offset = blockIdx.x * n / 2;

  float sum = 0.0f;
  float2 local_out_fp2[ite];
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = offset + col_id;
    local_out_fp2[i] = __half22float2(
        __hadd2(__hadd2(out_ptr[id], __ldg(&input_ptr[id])), __ldg(&bias_ptr[col_id])));
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((half2 *)out) + offset, n, s_, use_fp32);
}

template <const int ite>
__global__ void add_bias_input_layernorm_restore_output_v2(const float *out, const float *input,
                                                           const float *bias, const void *gamma,
                                                           const void *beta, int n, bool use_fp32,
                                                           float *out2, const int *batch_idx,
                                                           const int *word_idx,
                                                           const int seq_len) {
  const int batch_id = blockIdx.x / seq_len;
  const int seq_id = blockIdx.x % seq_len;
  const int batch_offset = __ldg(&batch_idx[batch_id]);
  const int batch_seq_len = __ldg(&batch_idx[batch_id + 1]) - batch_offset;
  int from_offset = (batch_offset + seq_id) * n;
  int offset = blockIdx.x * n;
  if (seq_id >= batch_seq_len) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + threadIdx.x;
      out2[offset + col_id] = 0.0f;
    }
    return;
  }

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = from_offset + col_id;
    local_out[i] = (float)(out[id] + input[id] + __ldg(&bias[col_id]));
    sum += local_out[i];
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out, sum, gamma, beta, out2 + offset, n, s_);
}

template <const int ite>
__global__ void add_bias_input_layernorm_restore_output_v2(const __half *out, const __half *input,
                                                           const __half *bias, const void *gamma,
                                                           const void *beta, int n, bool use_fp32,
                                                           __half *out2, const int *batch_idx,
                                                           const int *word_idx,
                                                           const int seq_len) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  const int batch_id = blockIdx.x / seq_len;
  const int seq_id = blockIdx.x % seq_len;
  const int batch_offset = __ldg(&batch_idx[batch_id]);
  const int batch_seq_len = __ldg(&batch_idx[batch_id + 1]) - batch_offset;
  int from_offset = (batch_offset + seq_id) * n / 2;
  int offset = blockIdx.x * n / 2;
  if (seq_id >= batch_seq_len) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + threadIdx.x;
      ((float *)out2)[offset + col_id] = 0.0f;
    }
    return;
  }

  float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = from_offset + col_id;
    local_out_fp2[i] = __half22float2(
        __hadd2(__hadd2(out_ptr[id], __ldg(&input_ptr[id])), __ldg(&bias_ptr[col_id])));
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((half2 *)out2) + offset, n, s_, use_fp32);
}

template <typename T>
void input_layernorm_kernel_launcher(T *output, const T *input, const void *gamma,
                                     const void *beta, int m, int n, int hidden_dim,
                                     cudaStream_t stream, bool use_fp32) {
  dim3 grid(m), block(hidden_dim);

  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      input_layernorm_v2<2>
          <<<grid, block.x / 2, 0, stream>>>(output, input, gamma, beta, n, use_fp32);
    else
      input_layernorm_v2<4>
          <<<grid, block.x / 4, 0, stream>>>(output, input, gamma, beta, n, use_fp32);
  } else
    input_layernorm<<<grid, block, 0, stream>>>(output, input, gamma, beta, n, use_fp32);
}

template <typename T>
void input_compress_layernorm_kernel_launcher(T *output, const T *input, const void *gamma,
                                              const void *beta, int m, int n, int hidden_dim,
                                              cudaStream_t stream, bool use_fp32, T *output2,
                                              int *batch_idx, int *word_idx) {
  dim3 grid(m), block(hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      input_compress_layernorm_v2<2><<<grid, block.x / 2, 0, stream>>>(
          output2, input, gamma, beta, n, use_fp32, output, batch_idx, word_idx);
    else
      input_compress_layernorm_v2<4><<<grid, block.x / 4, 0, stream>>>(
          output2, input, gamma, beta, n, use_fp32, output, batch_idx, word_idx);
  } else
    input_compress_layernorm<<<grid, block, 0, stream>>>(output2, input, gamma, beta, n, use_fp32,
                                                         output, batch_idx, word_idx);
}

template <typename T>
void add_bias_input_layernorm_kernel_launcher(T *output, const T *input, const T *bias,
                                              const void *gamma, const void *beta, int m, int n,
                                              int hidden_dim, cudaStream_t stream, bool use_fp32) {
  dim3 grid(m), block(hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      add_bias_input_layernorm_v2<2>
          <<<grid, block.x / 2, 0, stream>>>(output, input, bias, gamma, beta, n, use_fp32);
    else
      add_bias_input_layernorm_v2<4>
          <<<grid, block.x / 4, 0, stream>>>(output, input, bias, gamma, beta, n, use_fp32);
  } else {
    if (block.x < 32)
      block.x = 32;
    add_bias_input_layernorm<<<grid, block, 0, stream>>>(output, input, bias, gamma, beta, n,
                                                         use_fp32);
  }
}

template <typename T>
void add_bias_input_layernorm_restore_output_kernel_launcher(
    T *output, const T *input, const T *bias, const void *gamma, const void *beta, int m, int n,
    int hidden_dim, cudaStream_t stream, bool use_fp32, T *output2, int *batch_idx, int *word_idx,
    const int seq_len) {
  dim3 grid(m), block(hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      add_bias_input_layernorm_restore_output_v2<2><<<grid, block.x / 2, 0, stream>>>(
          output, input, bias, gamma, beta, n, use_fp32, output2, batch_idx, word_idx, seq_len);
    else
      add_bias_input_layernorm_restore_output_v2<4><<<grid, block.x / 4, 0, stream>>>(
          output, input, bias, gamma, beta, n, use_fp32, output2, batch_idx, word_idx, seq_len);
  } else
    add_bias_input_layernorm_restore_output<<<grid, block, 0, stream>>>(
        output, input, bias, gamma, beta, n, use_fp32, output2, batch_idx, word_idx, seq_len);
}

// ************** for pre_norm: out + bias + input -> out2, layernorm(out2) ->
// out ****************
template <typename T>
__global__ void add_bias_input_out_layernorm(T *out, const T *input, const T *bias, T *out2,
                                             const void *gamma, const void *beta, int n,
                                             bool use_fp32);

template <const int ite>
__global__ void add_bias_input_out_layernorm_v2(float *out, const float *input, const float *bias,
                                                float *out2, const void *gamma, const void *beta,
                                                int n, bool use_fp32) {
  int offset = blockIdx.x * n;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = offset + col_id;
    local_out[i] = out[id] + __ldg(&input[id]) + __ldg(&bias[col_id]);
    out2[id] = local_out[i];
    sum += local_out[i];
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out, sum, gamma, beta, out + offset, n, s_);
}

template <const int ite>
__global__ void add_bias_input_out_layernorm_v2(__half *out, const __half *input,
                                                const __half *bias, __half *out2,
                                                const void *gamma, const void *beta, int n,
                                                bool use_fp32) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;
  half2 *out2_ptr = (half2 *)out2;

  int offset = blockIdx.x * n / 2;

  float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = offset + col_id;
    half2 temp = __hadd2(__hadd2(out_ptr[id], __ldg(&input_ptr[id])), __ldg(&bias_ptr[col_id]));
    out2_ptr[id] = temp;
    local_out_fp2[i] = __half22float2(temp);
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((half2 *)out) + offset, n, s_, use_fp32);
}

template <typename T>
void add_bias_input_out_layernorm_kernel_launcher(T *output, const T *input, const T *bias,
                                                  T *output2, const void *gamma, const void *beta,
                                                  int m, int n, int hidden_dim,
                                                  cudaStream_t stream, bool use_fp32) {
  dim3 grid(m), block(hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      add_bias_input_out_layernorm_v2<2><<<grid, block.x / 2, 0, stream>>>(
          output, input, bias, output2, gamma, beta, n, use_fp32);
    else
      add_bias_input_out_layernorm_v2<4><<<grid, block.x / 4, 0, stream>>>(
          output, input, bias, output2, gamma, beta, n, use_fp32);
  } else
    add_bias_input_out_layernorm<<<grid, block, 0, stream>>>(output, input, bias, output2, gamma,
                                                             beta, n, use_fp32);
}

// ************** for conformer: (out + bias) * 0.5 + input -> out2,
// layernorm(out2) -> out ****************
template <typename T>
__global__ void add_bias_half_input_layernorm(T *out, const T *input, const T *bias,
                                              const void *gamma, const void *beta, int n,
                                              bool use_fp32);

template <const int ite>
__global__ void add_bias_half_input_layernorm_v2(float *out, const float *input, const float *bias,
                                                 const void *gamma, const void *beta, int n,
                                                 bool use_fp32) {
  int offset = blockIdx.x * n;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = offset + col_id;
    local_out[i] = (out[id] + __ldg(&bias[col_id])) * 0.5f + __ldg(&input[id]);
    sum += local_out[i];
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out, sum, gamma, beta, out + offset, n, s_);
}

template <const int ite>
__global__ void add_bias_half_input_layernorm_v2(__half *out, const __half *input,
                                                 const __half *bias, const void *gamma,
                                                 const void *beta, int n, bool use_fp32) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;
  int offset = blockIdx.x * n / 2;

  float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = offset + col_id;
    local_out_fp2[i] = __half22float2(
        __hadd2(__hmul2(__hadd2(out_ptr[id], __ldg(&bias_ptr[col_id])), half2(0.5f, 0.5f)),
                __ldg(&input_ptr[id])));
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((half2 *)out) + offset, n, s_, use_fp32);
}

template <typename T>
__global__ void add_bias_half_input_layernorm_restore_output(
    const T *out, const T *input, const T *bias, const void *gamma, const void *beta, int n,
    bool use_fp32, T *out2, const int *batch_idx, const int *word_idx, const int seq_len);

template <const int ite>
__global__ void add_bias_half_input_layernorm_restore_output_v2(
    const float *out, const float *input, const float *bias, const void *gamma, const void *beta,
    int n, bool use_fp32, float *out2, const int *batch_idx, const int *word_idx,
    const int seq_len) {
  const int batch_id = blockIdx.x / seq_len;
  const int seq_id = blockIdx.x % seq_len;
  const int batch_offset = __ldg(&batch_idx[batch_id]);
  const int batch_seq_len = __ldg(&batch_idx[batch_id + 1]) - batch_offset;
  int from_offset = (batch_offset + seq_id) * n;
  int offset = blockIdx.x * n;
  if (seq_id >= batch_seq_len) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + threadIdx.x;
      out2[offset + col_id] = 0.0f;
    }
    return;
  }

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = from_offset + col_id;
    local_out[i] = (out[id] + __ldg(&bias[col_id])) * 0.5f + __ldg(&input[id]);
    sum += local_out[i];
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out, sum, gamma, beta, out2 + offset, n, s_);
}

template <const int ite>
__global__ void add_bias_half_input_layernorm_restore_output_v2(
    const __half *out, const __half *input, const __half *bias, const void *gamma,
    const void *beta, int n, bool use_fp32, __half *out2, const int *batch_idx,
    const int *word_idx, const int seq_len) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  const int batch_id = blockIdx.x / seq_len;
  const int seq_id = blockIdx.x % seq_len;
  const int batch_offset = __ldg(&batch_idx[batch_id]);
  const int batch_seq_len = __ldg(&batch_idx[batch_id + 1]) - batch_offset;
  int from_offset = (batch_offset + seq_id) * n / 2;
  int offset = blockIdx.x * n / 2;
  if (seq_id >= batch_seq_len) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + threadIdx.x;
      ((float *)out2)[offset + col_id] = 0.0f;
    }
    return;
  }

  float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = from_offset + col_id;
    local_out_fp2[i] = __half22float2(
        __hadd2(__hmul2(__hadd2(out_ptr[id], __ldg(&bias_ptr[col_id])), half2(0.5f, 0.5f)),
                __ldg(&input_ptr[id])));
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((half2 *)out2) + offset, n, s_, use_fp32);
}

template <typename T>
__global__ void add_bias_half_input_out_layernorm(T *out, const T *input, const T *bias, T *out2,
                                                  const void *gamma, const void *beta, int n,
                                                  bool use_fp32);

template <const int ite>
__global__ void add_bias_half_input_out_layernorm_v2(float *out, const float *input,
                                                     const float *bias, float *out2,
                                                     const void *gamma, const void *beta, int n,
                                                     bool use_fp32) {
  int offset = blockIdx.x * n;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = offset + col_id;
    local_out[i] = (out[id] + __ldg(&bias[col_id])) * 0.5f + __ldg(&input[id]);
    out2[id] = local_out[i];
    sum += local_out[i];
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out, sum, gamma, beta, out + offset, n, s_);
}

template <const int ite>
__global__ void add_bias_half_input_out_layernorm_v2(__half *out, const __half *input,
                                                     const __half *bias, __half *out2,
                                                     const void *gamma, const void *beta, int n,
                                                     bool use_fp32) {
  half2 *out_ptr = (half2 *)out;
  half2 *out2_ptr = (half2 *)out2;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;
  int offset = blockIdx.x * n / 2;

  float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = offset + col_id;
    half2 temp =
        __hadd2(__hmul2(__hadd2(out_ptr[id], __ldg(&bias_ptr[col_id])), half2(0.5f, 0.5f)),
                __ldg(&input_ptr[id]));
    out2_ptr[id] = temp;
    local_out_fp2[i] = __half22float2(temp);
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((half2 *)out) + offset, n, s_, use_fp32);
}

template <typename T>
void add_bias_half_input_layernorm_kernel_launcher(T *output, const T *input, const T *bias,
                                                   const void *gamma, const void *beta, int m,
                                                   int n, int hidden_dim, cudaStream_t stream,
                                                   bool use_fp32) {
  dim3 grid(m), block(hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      add_bias_half_input_layernorm_v2<2>
          <<<grid, block.x / 2, 0, stream>>>(output, input, bias, gamma, beta, n, use_fp32);
    else
      add_bias_half_input_layernorm_v2<4>
          <<<grid, block.x / 4, 0, stream>>>(output, input, bias, gamma, beta, n, use_fp32);
  } else
    add_bias_half_input_layernorm<<<grid, block, 0, stream>>>(output, input, bias, gamma, beta, n,
                                                              use_fp32);
}

template <typename T>
void add_bias_half_input_layernorm_restore_output_kernel_launcher(
    T *output, const T *input, const T *bias, const void *gamma, const void *beta, int m, int n,
    int hidden_dim, cudaStream_t stream, bool use_fp32, T *output2, int *batch_idx, int *word_idx,
    const int seq_len) {
  dim3 grid(m), block(hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      add_bias_half_input_layernorm_restore_output_v2<2><<<grid, block.x / 2, 0, stream>>>(
          output, input, bias, gamma, beta, n, use_fp32, output2, batch_idx, word_idx, seq_len);
    else
      add_bias_half_input_layernorm_restore_output_v2<4><<<grid, block.x / 4, 0, stream>>>(
          output, input, bias, gamma, beta, n, use_fp32, output2, batch_idx, word_idx, seq_len);
  } else
    add_bias_half_input_layernorm_restore_output<<<grid, block, 0, stream>>>(
        output, input, bias, gamma, beta, n, use_fp32, output2, batch_idx, word_idx, seq_len);
}

template <typename T>
void add_bias_half_input_out_layernorm_kernel_launcher(T *output, const T *input, const T *bias,
                                                       T *output2, const void *gamma,
                                                       const void *beta, int m, int n,
                                                       int hidden_dim, cudaStream_t stream,
                                                       bool use_fp32) {
  dim3 grid(m), block(hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      add_bias_half_input_out_layernorm_v2<2><<<grid, block.x / 2, 0, stream>>>(
          output, input, bias, output2, gamma, beta, n, use_fp32);
    else
      add_bias_half_input_out_layernorm_v2<4><<<grid, block.x / 4, 0, stream>>>(
          output, input, bias, output2, gamma, beta, n, use_fp32);
  } else
    add_bias_half_input_out_layernorm<<<grid, block, 0, stream>>>(output, input, bias, output2,
                                                                  gamma, beta, n, use_fp32);
}
}  // namespace bytetransformer
