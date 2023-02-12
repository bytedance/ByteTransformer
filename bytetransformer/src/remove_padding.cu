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
#include "bytetransformer/include/remove_padding.h"

namespace bytetransformer {
template <>
__global__ void add_bias_input<float>(float *out, const float *input, const float *bias, int n) {
  int offset = blockIdx.x * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int index = offset + i;
    out[index] = out[index] + __ldg(&input[index]) + __ldg(&bias[i]);
  }
}

template <>
__global__ void add_bias_input<__half>(__half *out, const __half *input, const __half *bias,
                                       int n) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  int id = blockIdx.x * n / 2 + threadIdx.x;
  out_ptr[id] =
      __hadd2(__hadd2(out_ptr[id], __ldg(&input_ptr[id])), __ldg(&bias_ptr[threadIdx.x]));
}

template <>
__global__ void add_bias_input_restore_output<float>(const float *out, const float *input,
                                                     const float *bias, int n, float *out2,
                                                     const int *batch_idx, const int *word_idx,
                                                     const int seq_len) {
  const int batch_id = blockIdx.x / seq_len;
  const int seq_id = blockIdx.x % seq_len;
  const int batch_offset = __ldg(&batch_idx[batch_id]);
  const int batch_seq_len = __ldg(&batch_idx[batch_id + 1]) - batch_offset;
  int src_offset = (batch_offset + seq_id) * n;
  int dst_offset = blockIdx.x * n;
  if (seq_id >= batch_seq_len) {
    for (int i = threadIdx.x; i < n; i += blockDim.x)
      out2[dst_offset + i] = 0.0f;
    return;
  }

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int index = src_offset + i;
    out2[dst_offset + i] = out[index] + __ldg(&input[index]) + __ldg(&bias[i]);
  }
}

template <>
__global__ void add_bias_input_restore_output<__half>(const __half *out, const __half *input,
                                                      const __half *bias, int n, __half *out2,
                                                      const int *batch_idx, const int *word_idx,
                                                      const int seq_len) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

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

  ((half2 *)out2)[dst_offset] = __hadd2(
      __hadd2(out_ptr[src_offset], __ldg(&input_ptr[src_offset])), __ldg(&bias_ptr[threadIdx.x]));
}

template <>
__global__ void add_bias_half_input<float>(float *out, const float *input, const float *bias,
                                           int n) {
  int offset = blockIdx.x * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int index = offset + i;
    out[index] = __ldg(&input[index]) + (out[index] + __ldg(&bias[i])) * 0.5f;
  }
}

template <>
__global__ void add_bias_half_input<__half>(__half *out, const __half *input, const __half *bias,
                                            int n) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  int id = blockIdx.x * n / 2 + threadIdx.x;
  out_ptr[id] =
      __hadd2(__ldg(&input_ptr[id]),
              __hmul2(__hadd2(out_ptr[id], __ldg(&bias_ptr[threadIdx.x])), half2(0.5f, 0.5f)));
}

template <>
__global__ void add_bias_half_input_restore_output<float>(const float *out, const float *input,
                                                          const float *bias, int n, float *out2,
                                                          const int *batch_idx,
                                                          const int *word_idx, const int seq_len) {
  const int batch_id = blockIdx.x / seq_len;
  const int seq_id = blockIdx.x % seq_len;
  const int batch_offset = __ldg(&batch_idx[batch_id]);
  const int batch_seq_len = __ldg(&batch_idx[batch_id + 1]) - batch_offset;
  int src_offset = (batch_offset + seq_id) * n;
  int dst_offset = blockIdx.x * n;
  if (seq_id >= batch_seq_len) {
    for (int i = threadIdx.x; i < n; i += blockDim.x)
      out2[dst_offset + i] = 0.0f;
    return;
  }

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int index = src_offset + i;
    out2[dst_offset + i] = __ldg(&input[index]) + (__ldg(&out[index]) + __ldg(&bias[i])) * 0.5f;
  }
}

template <>
__global__ void add_bias_half_input_restore_output<__half>(const __half *out, const __half *input,
                                                           const __half *bias, int n, __half *out2,
                                                           const int *batch_idx,
                                                           const int *word_idx,
                                                           const int seq_len) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

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

  ((half2 *)out2)[dst_offset] =
      __hadd2(__ldg(&input_ptr[src_offset]),
              __hmul2(__hadd2(__ldg(&out_ptr[src_offset]), __ldg(&bias_ptr[threadIdx.x])),
                      half2(0.5f, 0.5f)));
}
}  // namespace bytetransformer
