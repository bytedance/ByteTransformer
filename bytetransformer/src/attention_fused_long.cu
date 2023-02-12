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
#include "bytetransformer/include/attention.h"
#include "bytetransformer/include/reduce.h"
#include <mma.h>

namespace bytetransformer {
#define SKEW_HALF 8  // offset for avoding bank conflict
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <const int max_seq_len, const int size_per_head, const int split_seq_len>
__global__ void wmma_attention_long_kernel(const half2 *qkv, const half2 *qkv_bias,
                                           const __half *attention_mask, __half *attention_output,
                                           const int seq_len, const float scale) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  extern __shared__ __half base[];
  __half(*s_kv)[size_per_head + SKEW_HALF] = (__half(*)[size_per_head + SKEW_HALF]) base;
  __half(*s_query)[size_per_head + SKEW_HALF] =
      (__half(*)[size_per_head + SKEW_HALF])(base + (max_seq_len) * (size_per_head + SKEW_HALF));
  __half(*s_logits)[max_seq_len + SKEW_HALF] = (__half(*)[max_seq_len + SKEW_HALF])(
      base + (split_seq_len + max_seq_len) * (size_per_head + SKEW_HALF));

  const int warpNums = (split_seq_len / 16) * (max_seq_len / 16);  //(blockDim.x  >> 5);
  const int warpId = (threadIdx.x >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = gridDim.x * (size_per_head / 2);
  const int thread_offset = blockIdx.x * (size_per_head / 2) + warp_tid;

  const int batch_seq_offset = blockIdx.z * seq_len;
  const int block_seq_len = min(split_seq_len, seq_len - (int)blockIdx.y * split_seq_len);
  const int batch_seq_block_offset = batch_seq_offset + blockIdx.y * split_seq_len;

  const int from_size = split_seq_len / 16;
  const int to_size = max_seq_len / 16;

  // loading Query
  half2 q_bias = __ldg(&qkv_bias[thread_offset]);
  for (int seq_id = warpId; seq_id < block_seq_len; seq_id += warpNums) {
    int pos = (batch_seq_block_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_query + offset) = __hadd2(__ldg(&qkv[pos]), q_bias);
  }

  // loading Key
  half2 k_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim]);
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_kv + offset) = __hadd2(__ldg(&qkv[pos + half_hidden_dim]), k_bias);
  }
  __syncthreads();

  if (warpId < from_size * to_size) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Q_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> K_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QK_mat;
    wmma::fill_fragment(QK_mat, 0.0f);
    const int warp_from_offset = (warpId / to_size) << 4;
    const int warp_to_offset = (warpId % to_size) << 4;

#pragma unroll
    for (int k = 0; k < 4; k++) {
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }
  __syncthreads();

  // softmax
  for (int from_id = warpId; from_id < block_seq_len; from_id += warpNums) {
    float max_val = -1e20f;

    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

#pragma unroll
    for (int i = 0; i < n; i++) {
      to_id[i] = warp_tid + (i << 5);
      logits[i] = -1e20f;

      if (to_id[i] < seq_len) {
        float mask =
            (float)__ldg(&attention_mask[(batch_seq_block_offset + from_id) * seq_len + to_id[i]]);
        mask = (1.0f - mask) * (-10000.0f);
        logits[i] = (float)(s_logits[from_id][to_id[i]]) * scale + mask;
      }
      max_val = max(max_val, logits[i]);
    }
    max_val = warpReduceMax(max_val);

    float sum_val = 0.0f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      logits[i] = __expf(logits[i] - max_val);
      sum_val += logits[i];
    }
    sum_val = warpReduceSum(sum_val) + 1e-6f;

#pragma unroll
    for (int i = 0; i < n; i++)
      if (to_id[i] < max_seq_len)
        s_logits[from_id][to_id[i]] = (__half)__fdividef(logits[i], sum_val);
  }

  // loading Value
  half2 v_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim * 2]);
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    ((__half2 *)(s_kv[seq_id]))[warp_tid] =
        __hadd2(__ldg(&qkv[pos + half_hidden_dim * 2]), v_bias);
  }

  // K dim clear 0
  for (int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  __syncthreads();

  //* V
  if (warpId < (from_size << 2)) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Logits_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QKV_mat;
    wmma::fill_fragment(QKV_mat, 0.0f);
    const int warp_from_offset = (warpId >> 2) << 4;
    const int warp_to_offset = (warpId & 0x3) * WMMA_K;

#pragma unroll
    for (int k = 0; k < to_size; k++) {
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  __syncthreads();

  for (int from_id = warpId; from_id < block_seq_len; from_id += warpNums) {
    int pos = (batch_seq_block_offset + from_id) * half_hidden_dim + thread_offset;
    ((__half2 *)(attention_output))[pos] = ((__half2 *)(s_query[from_id]))[warp_tid];
  }
#endif
}

template <const int max_seq_len, const int size_per_head, const int split_seq_len>
__global__ void wmma_attention_long_rm_kernel(const half2 *qkv, const half2 *qkv_bias,
                                              const __half *attention_mask,
                                              __half *attention_output, const float scale,
                                              const int *batch_idx) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  extern __shared__ __half base[];
  __half(*s_kv)[size_per_head + SKEW_HALF] = (__half(*)[size_per_head + SKEW_HALF]) base;
  __half(*s_query)[size_per_head + SKEW_HALF] =
      (__half(*)[size_per_head + SKEW_HALF])(base + (max_seq_len) * (size_per_head + SKEW_HALF));
  __half(*s_logits)[max_seq_len + SKEW_HALF] = (__half(*)[max_seq_len + SKEW_HALF])(
      base + (split_seq_len + max_seq_len) * (size_per_head + SKEW_HALF));

  const int warpNums = (split_seq_len / 16) * (max_seq_len / 16);  //(blockDim.x  >> 5);
  const int warpId = (threadIdx.x >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = gridDim.x * (size_per_head / 2);
  const int thread_offset = blockIdx.x * (size_per_head / 2) + warp_tid;

  // const int batch_seq_offset = blockIdx.z * seq_len;
  const int batch_seq_offset = __ldg(&batch_idx[blockIdx.z]);
  const int batch_seq_len = __ldg(&batch_idx[blockIdx.z + 1]) - batch_seq_offset;
  const int block_seq_len = min(split_seq_len, batch_seq_len - (int)blockIdx.y * split_seq_len);
  if (block_seq_len <= 0)
    return;
  const int batch_seq_len_pad = ((batch_seq_len + 15) >> 4) << 4;
  const int batch_seq_block_offset = batch_seq_offset + blockIdx.y * split_seq_len;

  const int from_size = split_seq_len / 16;
  const int to_size = max_seq_len / 16;

  // loading Query
  half2 q_bias = __ldg(&qkv_bias[thread_offset]);
  for (int seq_id = warpId; seq_id < block_seq_len; seq_id += warpNums) {
    int pos = (batch_seq_block_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_query + offset) = __hadd2(__ldg(&qkv[pos]), q_bias);
  }

  // loading Key
  half2 k_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim]);
  for (int seq_id = warpId; seq_id < batch_seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    *(__half2 *)(*s_kv + offset) = __hadd2(__ldg(&qkv[pos + half_hidden_dim]), k_bias);
  }
  __syncthreads();

  if (warpId < from_size * to_size) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Q_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> K_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QK_mat;
    wmma::fill_fragment(QK_mat, 0.0f);
    const int warp_from_offset = (warpId / to_size) << 4;
    const int warp_to_offset = (warpId % to_size) << 4;

    if (warp_to_offset < batch_seq_len_pad) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                               size_per_head + SKEW_HALF);
        wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K,
                               size_per_head + SKEW_HALF);
        wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
      }
      wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                              max_seq_len + SKEW_HALF, wmma::mem_row_major);
    }
  }
  __syncthreads();

  // softmax
  for (int from_id = warpId; from_id < block_seq_len; from_id += warpNums) {
    float max_val = -1e20f;

    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

#pragma unroll
    for (int i = 0; i < n; i++) {
      to_id[i] = warp_tid + (i << 5);
      logits[i] = to_id[i] < batch_seq_len ? (float)(s_logits[from_id][to_id[i]]) * scale : -1e20f;
      max_val = max(max_val, logits[i]);
    }
    max_val = warpReduceMax(max_val);

    float sum_val = 0.0f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      logits[i] = __expf(logits[i] - max_val);
      sum_val += logits[i];
    }
    sum_val = warpReduceSum(sum_val) + 1e-6f;

#pragma unroll
    for (int i = 0; i < n; i++)
      if (to_id[i] < batch_seq_len_pad)
        s_logits[from_id][to_id[i]] = (__half)__fdividef(logits[i], sum_val);
  }

  // loading Value
  half2 v_bias = __ldg(&qkv_bias[thread_offset + half_hidden_dim * 2]);
  for (int seq_id = warpId; seq_id < batch_seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    ((__half2 *)(s_kv[seq_id]))[warp_tid] =
        __hadd2(__ldg(&qkv[pos + half_hidden_dim * 2]), v_bias);
  }

  // K dim clear 0
  for (int seq_id = batch_seq_len + warpId; seq_id < batch_seq_len_pad; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  __syncthreads();

  //* V
  if (warpId < (from_size << 2)) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Logits_mat;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> V_mat;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> QKV_mat;
    wmma::fill_fragment(QKV_mat, 0.0f);
    const int warp_from_offset = (warpId >> 2) << 4;
    const int warp_to_offset = (warpId & 0x3) * WMMA_K;

#pragma unroll
    for (int k = 0; k < (batch_seq_len_pad / 16); k++) {
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  __syncthreads();

  for (int from_id = warpId; from_id < block_seq_len; from_id += warpNums) {
    int pos = (batch_seq_block_offset + from_id) * half_hidden_dim + thread_offset;
    ((__half2 *)(attention_output))[pos] = ((__half2 *)(s_query[from_id]))[warp_tid];
  }
#endif
}

// __shared__ __half     s_kv  [max_seq_len][size_per_head + SKEW_HALF];
// __shared__ __half  s_query[split_seq_len][size_per_head + SKEW_HALF];
// __shared__ __half s_logits[split_seq_len][max_seq_len   + SKEW_HALF];

#define WMMA_ATTENTION_LONG(SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN)                                    \
  shared_memory_size =                                                                            \
      ((SEQ_LEN + SPLIT_LEN) * (SIZE_PER_HEAD + SKEW_HALF) + SPLIT_LEN * (SEQ_LEN + SKEW_HALF)) * \
      2;                                                                                          \
  if (shared_memory_size > 48 * 1024)                                                             \
    cudaFuncSetAttribute(wmma_attention_long_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>,           \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);                 \
  grid.x = head_num_, grid.y = (SEQ_LEN + SPLIT_LEN - 1) / SPLIT_LEN, grid.z = batch_size,        \
  block.x = 32 * (SPLIT_LEN / 16 * split_count);                                                  \
  wmma_attention_long_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>                                   \
      <<<grid, block, shared_memory_size, infer_param.stream>>>(                                  \
          qkv_ptr, qkv_bias_ptr, (__half *)atten_mask, (__half *)attention_output, seq_len,       \
          scale)

#define WMMA_ATTENTION_LONG_RM(SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN)                                 \
  shared_memory_size =                                                                            \
      ((SEQ_LEN + SPLIT_LEN) * (SIZE_PER_HEAD + SKEW_HALF) + SPLIT_LEN * (SEQ_LEN + SKEW_HALF)) * \
      2;                                                                                          \
  if (shared_memory_size > 48 * 1024)                                                             \
    cudaFuncSetAttribute(wmma_attention_long_rm_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>,        \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);                 \
  grid.x = head_num_, grid.y = (SEQ_LEN + SPLIT_LEN - 1) / SPLIT_LEN, grid.z = batch_size,        \
  block.x = 32 * (SPLIT_LEN / 16 * split_count);                                                  \
  wmma_attention_long_rm_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>                                \
      <<<grid, block, shared_memory_size, infer_param.stream>>>(                                  \
          qkv_ptr, qkv_bias_ptr, (__half *)atten_mask, (__half *)attention_output, scale,         \
          et_param.batch_idx)

template <OperationType OpType>
void Attention<OpType>::fused_long_infer(AttentionInferParam infer_param) {
  const DataType_ *atten_mask = infer_param.atten_mask;
  DataType_ *attention_output = infer_param.attention_output;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;

  if (OpType == OperationType::HALF) {
    const half2 *qkv_ptr = (const half2 *)infer_param.qkv;
    const half2 *qkv_bias_ptr = (const half2 *)param_.attr_bias_QKV;

    float scale = (1.0f / sqrt(size_per_head_ * 1.0f)) / param_.tao;

    dim3 grid, block;
    int shared_memory_size = 0;
    const int split_count = (seq_len + 15) / 16;
    switch (split_count) {
      case 6:
        WMMA_ATTENTION_LONG(96, 64, 48);
        break;  //  80 < seq_len <=  96
      case 7:
        WMMA_ATTENTION_LONG(112, 64, 64);
        break;  //  96 < seq_len <= 112
      case 8:
        WMMA_ATTENTION_LONG(128, 64, 64);
        break;  // 112 < seq_len <= 128
      case 9:
        WMMA_ATTENTION_LONG(144, 64, 48);
        break;  // 128 < seq_len <= 144
      case 10:
        WMMA_ATTENTION_LONG(160, 64, 48);
        break;  // 144 < seq_len <= 160
      case 11:
        WMMA_ATTENTION_LONG(176, 64, 32);
        break;  // 160 < seq_len <= 176
      case 12:
        WMMA_ATTENTION_LONG(192, 64, 32);
        break;  // 176 < seq_len <= 192
      case 13:
        WMMA_ATTENTION_LONG(208, 64, 32);
        break;  // 192 < seq_len <= 208
      case 14:
        WMMA_ATTENTION_LONG(224, 64, 32);
        break;  // 208 < seq_len <= 224
      case 15:
        WMMA_ATTENTION_LONG(240, 64, 32);
        break;  // 224 < seq_len <= 240
      case 16:
        WMMA_ATTENTION_LONG(256, 64, 32);
        break;  // 240 < seq_len <= 256
      case 17:
        WMMA_ATTENTION_LONG(272, 64, 16);
        break;  // 256 < seq_len <= 272
      case 18:
        WMMA_ATTENTION_LONG(288, 64, 16);
        break;  // 272 < seq_len <= 288
      case 19:
        WMMA_ATTENTION_LONG(304, 64, 16);
        break;  // 288 < seq_len <= 304
      case 20:
        WMMA_ATTENTION_LONG(320, 64, 16);
        break;  // 304 < seq_len <= 320
      case 21:
        WMMA_ATTENTION_LONG(336, 64, 16);
        break;  // 320 < seq_len <= 336
      case 22:
        WMMA_ATTENTION_LONG(352, 64, 16);
        break;  // 336 < seq_len <= 352
    }
  }
}

template <OperationType OpType>
void Attention<OpType>::fused_long_rm_infer(AttentionInferParam infer_param) {
  const DataType_ *atten_mask = infer_param.atten_mask;
  DataType_ *attention_output = infer_param.attention_output;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;
  ET_Param et_param = infer_param.et_param;

  if (OpType == OperationType::HALF) {
    const half2 *qkv_ptr = (const half2 *)infer_param.qkv;
    const half2 *qkv_bias_ptr = (const half2 *)param_.attr_bias_QKV;
    float scale = (1.0f / sqrt(size_per_head_ * 1.0f)) / param_.tao;

    dim3 grid, block;
    int shared_memory_size = 0;
    const int split_count = (seq_len + 15) / 16;
    switch (split_count) {
      case 6:
        WMMA_ATTENTION_LONG_RM(96, 64, 48);
        break;  //  80 < seq_len <=  96
      case 7:
        WMMA_ATTENTION_LONG_RM(112, 64, 64);
        break;  //  96 < seq_len <= 112
      case 8:
        WMMA_ATTENTION_LONG_RM(128, 64, 64);
        break;  // 112 < seq_len <= 128
      case 9:
        WMMA_ATTENTION_LONG_RM(144, 64, 48);
        break;  // 128 < seq_len <= 144
      case 10:
        WMMA_ATTENTION_LONG_RM(160, 64, 48);
        break;  // 144 < seq_len <= 160
      case 11:
        WMMA_ATTENTION_LONG_RM(176, 64, 32);
        break;  // 160 < seq_len <= 176
      case 12:
        WMMA_ATTENTION_LONG_RM(192, 64, 32);
        break;  // 176 < seq_len <= 192
      case 13:
        WMMA_ATTENTION_LONG_RM(208, 64, 32);
        break;  // 192 < seq_len <= 208
      case 14:
        WMMA_ATTENTION_LONG_RM(224, 64, 32);
        break;  // 208 < seq_len <= 224
      case 15:
        WMMA_ATTENTION_LONG_RM(240, 64, 32);
        break;  // 224 < seq_len <= 240
      case 16:
        WMMA_ATTENTION_LONG_RM(256, 64, 32);
        break;  // 240 < seq_len <= 256
      case 17:
        WMMA_ATTENTION_LONG_RM(272, 64, 16);
        break;  // 256 < seq_len <= 272
      case 18:
        WMMA_ATTENTION_LONG_RM(288, 64, 16);
        break;  // 272 < seq_len <= 288
      case 19:
        WMMA_ATTENTION_LONG_RM(304, 64, 16);
        break;  // 288 < seq_len <= 304
      case 20:
        WMMA_ATTENTION_LONG_RM(320, 64, 16);
        break;  // 304 < seq_len <= 320
      case 21:
        WMMA_ATTENTION_LONG_RM(336, 64, 16);
        break;  // 320 < seq_len <= 336
      case 22:
        WMMA_ATTENTION_LONG_RM(352, 64, 16);
        break;  // 336 < seq_len <= 352
    }
  }
}

template void Attention<OperationType::FP32>::fused_long_infer(AttentionInferParam infer_param);
template void Attention<OperationType::HALF>::fused_long_infer(AttentionInferParam infer_param);
template void Attention<OperationType::FP32>::fused_long_rm_infer(AttentionInferParam infer_param);
template void Attention<OperationType::HALF>::fused_long_rm_infer(AttentionInferParam infer_param);
}  // namespace bytetransformer
