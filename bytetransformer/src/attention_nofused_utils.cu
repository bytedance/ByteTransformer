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
#include "bytetransformer/include/attention_nofused_utils.h"

namespace bytetransformer {
template <>
__global__ void add_QKV_bias<float>(const float *QKV, const float *bias_QKV, float *q_buf,
                                    float *k_buf, float *v_buf, const int batch_size,
                                    const int seq_len, const int head_num,
                                    const int half_size_per_head, const bool is_roformer) {
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.x / half_size_per_head;
  int id = threadIdx.x % half_size_per_head;
  int src_id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * 3) + threadIdx.x;
  int trt_id = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half_size_per_head + id;

  float2 q_value = ((float2 *)QKV)[src_id], q_bias = __ldg(&((float2 *)bias_QKV)[threadIdx.x]);
  float2 k_value = ((float2 *)QKV)[src_id + blockDim.x],
         k_bias = __ldg(&((float2 *)bias_QKV)[threadIdx.x + blockDim.x]);
  float2 v_value = ((float2 *)QKV)[src_id + blockDim.x * 2],
         v_bias = __ldg(&((float2 *)bias_QKV)[threadIdx.x + blockDim.x * 2]);
  q_value.x += q_bias.x, q_value.y += q_bias.y;
  k_value.x += k_bias.x, k_value.y += k_bias.y;
  v_value.x += v_bias.x, v_value.y += v_bias.y;

  if (is_roformer) {
    float2 ro_q = make_float2(-q_value.y, q_value.x);
    float2 ro_k = make_float2(-k_value.y, k_value.x);
    float position_enc = __fdividef(seq_id, __powf(10000.0f, __fdividef(id, half_size_per_head)));
    float sin_pos = __sinf(position_enc);
    float cos_pos = __cosf(position_enc);
    q_value.x = q_value.x * cos_pos + ro_q.x * sin_pos,
    q_value.y = q_value.y * cos_pos + ro_q.y * sin_pos;
    k_value.x = k_value.x * cos_pos + ro_k.x * sin_pos,
    k_value.y = k_value.y * cos_pos + ro_k.y * sin_pos;
  }

  ((float2 *)q_buf)[trt_id] = q_value;
  ((float2 *)k_buf)[trt_id] = k_value;
  ((float2 *)v_buf)[trt_id] = v_value;
}

template <>
__global__ void add_QKV_bias<__half>(const __half *QKV, const __half *bias_QKV, __half *q_buf,
                                     __half *k_buf, __half *v_buf, const int batch_size,
                                     const int seq_len, const int head_num,
                                     const int half_size_per_head, const bool is_roformer) {
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.x / half_size_per_head;
  int id = threadIdx.x % half_size_per_head;
  int src_id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * 3) + threadIdx.x;
  int trt_id = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half_size_per_head + id;
  half2 q_value =
      __hadd2(((const half2 *)QKV)[src_id], __ldg(&((const half2 *)bias_QKV)[threadIdx.x]));
  half2 k_value = __hadd2(((const half2 *)QKV)[src_id + blockDim.x],
                          __ldg(&((const half2 *)bias_QKV)[threadIdx.x + blockDim.x]));
  half2 v_value = __hadd2(((const half2 *)QKV)[src_id + blockDim.x * 2],
                          __ldg(&((const half2 *)bias_QKV)[threadIdx.x + blockDim.x * 2]));

  if (is_roformer) {
    half2 ro_q = half2(-q_value.y, q_value.x);
    half2 ro_k = half2(-k_value.y, k_value.x);
    float position_enc = __fdividef(seq_id, __powf(10000.0f, __fdividef(id, half_size_per_head)));
    half2 sin_pos = __float2half2_rn(__sinf(position_enc));
    half2 cos_pos = __float2half2_rn(__cosf(position_enc));
    q_value = __hadd2(__hmul2(q_value, cos_pos), __hmul2(ro_q, sin_pos));
    k_value = __hadd2(__hmul2(k_value, cos_pos), __hmul2(ro_k, sin_pos));
  }

  ((half2 *)q_buf)[trt_id] = q_value;
  ((half2 *)k_buf)[trt_id] = k_value;
  ((half2 *)v_buf)[trt_id] = v_value;
}

template <>
__global__ void add_QKV_bias_padding<float>(const float *QKV, const float *bias_QKV, float *q_buf,
                                            float *k_buf, float *v_buf, const int batch_size,
                                            const int seq_len, const int head_num,
                                            const int half_size_per_head, const bool is_roformer,
                                            const int *batch_idx, const int *word_idx) {
  const int batch_id = blockIdx.y;
  const int seq_id = blockIdx.x;
  int head_id = threadIdx.x / half_size_per_head;
  int id = threadIdx.x % half_size_per_head;
  int trt_id = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half_size_per_head + id;
  const int batch_offset = __ldg(&batch_idx[blockIdx.y]);
  const int batch_seq_len = __ldg(&batch_idx[blockIdx.y + 1]) - batch_offset;
  if (seq_id < batch_seq_len) {
    int src_id = (batch_offset + seq_id) * (blockDim.x * 3) + threadIdx.x;
    float2 q_value = ((float2 *)QKV)[src_id], q_bias = __ldg(&((float2 *)bias_QKV)[threadIdx.x]);
    float2 k_value = ((float2 *)QKV)[src_id + blockDim.x],
           k_bias = __ldg(&((float2 *)bias_QKV)[threadIdx.x + blockDim.x]);
    float2 v_value = ((float2 *)QKV)[src_id + blockDim.x * 2],
           v_bias = __ldg(&((float2 *)bias_QKV)[threadIdx.x + blockDim.x * 2]);
    q_value.x += q_bias.x, q_value.y += q_bias.y;
    k_value.x += k_bias.x, k_value.y += k_bias.y;
    v_value.x += v_bias.x, v_value.y += v_bias.y;

    if (is_roformer) {
      float2 ro_q = make_float2(-q_value.y, q_value.x);
      float2 ro_k = make_float2(-k_value.y, k_value.x);
      float position_enc =
          __fdividef(seq_id, __powf(10000.0f, __fdividef(id, half_size_per_head)));
      float sin_pos = __sinf(position_enc);
      float cos_pos = __cosf(position_enc);
      q_value.x = q_value.x * cos_pos + ro_q.x * sin_pos,
      q_value.y = q_value.y * cos_pos + ro_q.y * sin_pos;
      k_value.x = k_value.x * cos_pos + ro_k.x * sin_pos,
      k_value.y = k_value.y * cos_pos + ro_k.y * sin_pos;
    }

    ((float2 *)q_buf)[trt_id] = q_value;
    ((float2 *)k_buf)[trt_id] = k_value;
    ((float2 *)v_buf)[trt_id] = v_value;
  } else {
    float2 zero = make_float2(0.0f, 0.0f);
    ((float2 *)q_buf)[trt_id] = zero;
    ((float2 *)k_buf)[trt_id] = zero;
    ((float2 *)v_buf)[trt_id] = zero;
  }
}

template <>
__global__ void add_QKV_bias_padding<__half>(const __half *QKV, const __half *bias_QKV,
                                             __half *q_buf, __half *k_buf, __half *v_buf,
                                             const int batch_size, const int seq_len,
                                             const int head_num, const int half_size_per_head,
                                             const bool is_roformer, const int *batch_idx,
                                             const int *word_idx) {
  const int batch_id = blockIdx.y;
  const int seq_id = blockIdx.x;
  int head_id = threadIdx.x / half_size_per_head;
  int id = threadIdx.x % half_size_per_head;
  int trt_id = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half_size_per_head + id;
  const int batch_offset = __ldg(&batch_idx[blockIdx.y]);
  const int batch_seq_len = __ldg(&batch_idx[blockIdx.y + 1]) - batch_offset;
  if (seq_id < batch_seq_len) {
    int src_id = (batch_offset + seq_id) * (blockDim.x * 3) + threadIdx.x;
    half2 q_value =
        __hadd2(((const half2 *)QKV)[src_id], __ldg(&((const half2 *)bias_QKV)[threadIdx.x]));
    half2 k_value = __hadd2(((const half2 *)QKV)[src_id + blockDim.x],
                            __ldg(&((const half2 *)bias_QKV)[threadIdx.x + blockDim.x]));
    half2 v_value = __hadd2(((const half2 *)QKV)[src_id + blockDim.x * 2],
                            __ldg(&((const half2 *)bias_QKV)[threadIdx.x + blockDim.x * 2]));

    if (is_roformer) {
      half2 ro_q = half2(-q_value.y, q_value.x);
      half2 ro_k = half2(-k_value.y, k_value.x);
      float position_enc =
          __fdividef(seq_id, __powf(10000.0f, __fdividef(id, half_size_per_head)));
      half2 sin_pos = __float2half2_rn(__sinf(position_enc));
      half2 cos_pos = __float2half2_rn(__cosf(position_enc));
      q_value = __hadd2(__hmul2(q_value, cos_pos), __hmul2(ro_q, sin_pos));
      k_value = __hadd2(__hmul2(k_value, cos_pos), __hmul2(ro_k, sin_pos));
    }

    ((half2 *)q_buf)[trt_id] = q_value;
    ((half2 *)k_buf)[trt_id] = k_value;
    ((half2 *)v_buf)[trt_id] = v_value;
  } else {
    ((float *)q_buf)[trt_id] = 0.0f;
    ((float *)k_buf)[trt_id] = 0.0f;
    ((float *)v_buf)[trt_id] = 0.0f;
  }
}

template <>
__global__ void transpose<float>(const float *src, float *dst, const int batch_size,
                                 const int seq_len, const int head_num, const int size_per_head) {
  int batch_id = blockIdx.x / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int head_id = threadIdx.y;
  int src_offset =
      ((head_id * batch_size + batch_id) * seq_len + seq_id) * size_per_head + threadIdx.x;
  int dst_offset = (blockIdx.x * head_num + head_id) * size_per_head + threadIdx.x;
  dst[dst_offset] = src[src_offset];
}

template <>
__global__ void transpose<__half>(const __half *src, __half *dst, const int batch_size,
                                  const int seq_len, const int head_num, const int size_per_head) {
  int batch_id = blockIdx.x / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int head_id = threadIdx.y;
  int src_offset =
      ((head_id * batch_size + batch_id) * seq_len + seq_id) * size_per_head + threadIdx.x;
  int dst_offset = (blockIdx.x * head_num + head_id) * size_per_head + threadIdx.x;
  ((half2 *)dst)[dst_offset] = ((const half2 *)src)[src_offset];
}

template <>
__global__ void transpose_rm_padding<float>(const float *src, float *dst, const int batch_size,
                                            const int seq_len, const int head_num,
                                            const int size_per_head, const int *batch_idx,
                                            const int *word_idx) {
  int offset = word_idx[blockIdx.x];
  int batch_id = offset / seq_len;  // batch_idx[blockIdx.x]
  int seq_id = offset % seq_len;    // word_idx[blockIdx.x]
  int head_id = threadIdx.y;
  int src_offset =
      ((head_id * batch_size + batch_id) * seq_len + seq_id) * size_per_head + threadIdx.x;
  int dst_offset = (blockIdx.x * head_num + head_id) * size_per_head + threadIdx.x;
  dst[dst_offset] = src[src_offset];
}

template <>
__global__ void transpose_rm_padding<__half>(const __half *src, __half *dst, const int batch_size,
                                             const int seq_len, const int head_num,
                                             const int size_per_head, const int *batch_idx,
                                             const int *word_idx) {
  int offset = word_idx[blockIdx.x];
  int batch_id = offset / seq_len;  // batch_idx[blockIdx.x]
  int seq_id = offset % seq_len;    // word_idx[blockIdx.x]
  int head_id = threadIdx.y;
  int src_offset =
      ((head_id * batch_size + batch_id) * seq_len + seq_id) * size_per_head + threadIdx.x;
  int dst_offset = (blockIdx.x * head_num + head_id) * size_per_head + threadIdx.x;
  ((half2 *)dst)[dst_offset] = ((const half2 *)src)[src_offset];
}
}  // namespace bytetransformer
