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
#include "bytetransformer/include/layernorm.h"

namespace bytetransformer {
template <>
__global__ void input_layernorm<float>(float *out, const float *input, const void *gamma,
                                       const void *beta, int n, bool use_fp32) {
  int offset = blockIdx.x * n + threadIdx.x;

  float local_out = __ldg(&input[offset]);

  __shared__ float s_[2];  // s_mean & s_variance
  layernorm(local_out, gamma, beta, out + offset, n, s_);
}

template <>
__global__ void input_layernorm<__half>(__half *out, const __half *input, const void *gamma,
                                        const void *beta, int n, bool use_fp32) {
  int offset = blockIdx.x * n / 2 + threadIdx.x;

  half2 local_out((__half)0.0f, (__half)0.0f);
  if (threadIdx.x < n / 2)
    local_out = __ldg(&((const half2 *)input)[offset]);

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, ((half2 *)out) + offset, n, s_, use_fp32);
}

template <>
__global__ void input_compress_layernorm<float>(float *out, const float *input, const void *gamma,
                                                const void *beta, int n, bool use_fp32,
                                                float *out2, const int *batch_idx,
                                                const int *word_idx) {
  int src_offset = __ldg(&word_idx[blockIdx.x]) * n + threadIdx.x;
  int dst_offset = blockIdx.x * n + threadIdx.x;

  float local_out = __ldg(&input[src_offset]);
  out[dst_offset] = local_out;

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, out2 + dst_offset, n, s_);
}

template <>
__global__ void input_compress_layernorm<__half>(__half *out, const __half *input,
                                                 const void *gamma, const void *beta, int n,
                                                 bool use_fp32, __half *out2, const int *batch_idx,
                                                 const int *word_idx) {
  int src_offset = __ldg(&word_idx[blockIdx.x]) * n / 2 + threadIdx.x;
  int dst_offset = blockIdx.x * n / 2 + threadIdx.x;

  half2 local_out = __ldg(&((const half2 *)input)[src_offset]);
  ((half2 *)out)[dst_offset] = local_out;

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, ((half2 *)out2) + dst_offset, n, s_, use_fp32);
}

template <>
__global__ void add_bias_input_layernorm<float>(float *out, const float *input, const float *bias,
                                                const void *gamma, const void *beta, int n,
                                                bool use_fp32) {
  int offset = blockIdx.x * n + threadIdx.x;

  float local_out = (float)(out[offset] + __ldg(&input[offset]) + __ldg(&bias[threadIdx.x]));

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, out + offset, n, s_);
}

template <>
__global__ void add_bias_input_layernorm<__half>(__half *out, const __half *input,
                                                 const __half *bias, const void *gamma,
                                                 const void *beta, int n, bool use_fp32) {
  int offset = blockIdx.x * n / 2 + threadIdx.x;

  half2 local_out((__half)0.0f, (__half)0.0f);
  if (threadIdx.x < n / 2)
    local_out = __hadd2(__hadd2(((half2 *)out)[offset], __ldg(&((const half2 *)input)[offset])),
                        __ldg(&((const half2 *)bias)[threadIdx.x]));

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, ((half2 *)out) + offset, n, s_, use_fp32);
}

template <>
__global__ void add_bias_input_layernorm_restore_output<float>(
    const float *out, const float *input, const float *bias, const void *gamma, const void *beta,
    int n, bool use_fp32, float *out2, const int *batch_idx, const int *word_idx,
    const int seq_len) {
  const int batch_id = blockIdx.x / seq_len;
  const int seq_id = blockIdx.x % seq_len;
  const int batch_offset = __ldg(&batch_idx[batch_id]);
  const int batch_seq_len = __ldg(&batch_idx[batch_id + 1]) - batch_offset;
  int src_offset = (batch_offset + seq_id) * n + threadIdx.x;
  int dst_offset = blockIdx.x * n + threadIdx.x;
  if (seq_id >= batch_seq_len) {
    out2[dst_offset] = 0.0f;
    return;
  }

  float local_out =
      (float)(__ldg(&out[src_offset]) + __ldg(&input[src_offset]) + __ldg(&bias[threadIdx.x]));
  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, out2 + dst_offset, n, s_);
}

template <>
__global__ void add_bias_input_layernorm_restore_output<__half>(
    const __half *out, const __half *input, const __half *bias, const void *gamma,
    const void *beta, int n, bool use_fp32, __half *out2, const int *batch_idx,
    const int *word_idx, const int seq_len) {
  const int batch_id = blockIdx.x / seq_len;
  const int seq_id = blockIdx.x % seq_len;
  const int batch_offset = __ldg(&batch_idx[batch_id]);
  const int batch_seq_len = __ldg(&batch_idx[batch_id + 1]) - batch_offset;
  int src_offset = (batch_offset + seq_id) * n / 2 + threadIdx.x;
  int dst_offset = blockIdx.x * n / 2 + threadIdx.x;
  if (seq_id >= batch_seq_len) {
    ((float *)out2)[dst_offset] = 0.0f;
    return;
  }

  half2 local_out = __hadd2(
      __hadd2(__ldg(&((half2 *)out)[src_offset]), __ldg(&((const half2 *)input)[src_offset])),
      __ldg(&((const half2 *)bias)[threadIdx.x]));

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, ((half2 *)out2) + dst_offset, n, s_, use_fp32);
}

// ************** for pre_norm: out + bias + input -> out2, layernorm(out2) -> out ****************
template <>
__global__ void add_bias_input_out_layernorm<float>(float *out, const float *input,
                                                    const float *bias, float *out2,
                                                    const void *gamma, const void *beta, int n,
                                                    bool use_fp32) {
  int offset = blockIdx.x * n + threadIdx.x;

  float local_out = out[offset] + __ldg(&input[offset]) + __ldg(&bias[threadIdx.x]);
  out2[offset] = local_out;

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, out + offset, n, s_);
}

template <>
__global__ void add_bias_input_out_layernorm<__half>(__half *out, const __half *input,
                                                     const __half *bias, __half *out2,
                                                     const void *gamma, const void *beta, int n,
                                                     bool use_fp32) {
  int offset = blockIdx.x * n / 2 + threadIdx.x;

  half2 local_out =
      __hadd2(__hadd2(((half2 *)out)[offset], __ldg(&((const half2 *)input)[offset])),
              __ldg(&((const half2 *)bias)[threadIdx.x]));
  ((half2 *)out2)[offset] = local_out;

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, ((half2 *)out) + offset, n, s_, use_fp32);
}

// ************** for conformer: (out + bias) * 0.5 + input -> out2, layernorm(out2) -> out
// ****************
template <>
__global__ void add_bias_half_input_layernorm<float>(float *out, const float *input,
                                                     const float *bias, const void *gamma,
                                                     const void *beta, int n, bool use_fp32) {
  int offset = blockIdx.x * n + threadIdx.x;

  float local_out = (out[offset] + __ldg(&bias[threadIdx.x])) * 0.5f + __ldg(&input[offset]);

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, out + offset, n, s_);
}

template <>
__global__ void add_bias_half_input_layernorm<__half>(__half *out, const __half *input,
                                                      const __half *bias, const void *gamma,
                                                      const void *beta, int n, bool use_fp32) {
  int offset = blockIdx.x * n / 2 + threadIdx.x;

  half2 local_out =
      __hadd2(__hmul2(__hadd2(((half2 *)out)[offset], __ldg(&((const half2 *)bias)[threadIdx.x])),
                      half2(0.5f, 0.5f)),
              __ldg(&((const half2 *)input)[offset]));

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, ((half2 *)out) + offset, n, s_, use_fp32);
}

template <>
__global__ void add_bias_half_input_layernorm_restore_output<float>(
    const float *out, const float *input, const float *bias, const void *gamma, const void *beta,
    int n, bool use_fp32, float *out2, const int *batch_idx, const int *word_idx,
    const int seq_len) {
  const int batch_id = blockIdx.x / seq_len;
  const int seq_id = blockIdx.x % seq_len;
  const int batch_offset = __ldg(&batch_idx[batch_id]);
  const int batch_seq_len = __ldg(&batch_idx[batch_id + 1]) - batch_offset;
  int src_offset = (batch_offset + seq_id) * n + threadIdx.x;
  int dst_offset = blockIdx.x * n + threadIdx.x;
  if (seq_id >= batch_seq_len) {
    out2[dst_offset] = 0.0f;
    return;
  }

  float local_out =
      (__ldg(&out[src_offset]) + __ldg(&bias[threadIdx.x])) * 0.5f + __ldg(&input[src_offset]);

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, out2 + dst_offset, n, s_);
}

template <>
__global__ void add_bias_half_input_layernorm_restore_output<__half>(
    const __half *out, const __half *input, const __half *bias, const void *gamma,
    const void *beta, int n, bool use_fp32, __half *out2, const int *batch_idx,
    const int *word_idx, const int seq_len) {
  const int batch_id = blockIdx.x / seq_len;
  const int seq_id = blockIdx.x % seq_len;
  const int batch_offset = __ldg(&batch_idx[batch_id]);
  const int batch_seq_len = __ldg(&batch_idx[batch_id + 1]) - batch_offset;
  int src_offset = (batch_offset + seq_id) * n / 2 + threadIdx.x;
  int dst_offset = blockIdx.x * n / 2 + threadIdx.x;
  if (seq_id >= batch_seq_len) {
    ((float *)out2)[dst_offset] = 0.0f;
    return;
  }

  half2 local_out = __hadd2(__hmul2(__hadd2(__ldg(&((half2 *)out)[src_offset]),
                                            __ldg(&((const half2 *)bias)[threadIdx.x])),
                                    half2(0.5f, 0.5f)),
                            __ldg(&((const half2 *)input)[src_offset]));

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, ((half2 *)out2) + dst_offset, n, s_, use_fp32);
}

template <>
__global__ void add_bias_half_input_out_layernorm<float>(float *out, const float *input,
                                                         const float *bias, float *out2,
                                                         const void *gamma, const void *beta,
                                                         int n, bool use_fp32) {
  int offset = blockIdx.x * n + threadIdx.x;

  float local_out = (out[offset] + __ldg(&bias[threadIdx.x])) * 0.5f + __ldg(&input[offset]);
  out2[offset] = local_out;

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, out + offset, n, s_);
}

template <>
__global__ void add_bias_half_input_out_layernorm<__half>(__half *out, const __half *input,
                                                          const __half *bias, __half *out2,
                                                          const void *gamma, const void *beta,
                                                          int n, bool use_fp32) {
  int offset = blockIdx.x * n / 2 + threadIdx.x;

  half2 local_out =
      __hadd2(__hmul2(__hadd2(((half2 *)out)[offset], __ldg(&((const half2 *)bias)[threadIdx.x])),
                      half2(0.5f, 0.5f)),
              __ldg(&((const half2 *)input)[offset]));
  ((half2 *)out2)[offset] = local_out;

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, ((half2 *)out) + offset, n, s_, use_fp32);
}
}  // namespace bytetransformer
