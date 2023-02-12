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
// ************************** build_sequence_length_padding_offset **************************
template <typename T>
__inline__ __device__ T warpPrefixSum(int id, T count) {
  for (int i = 1; i < 32; i <<= 1) {
    T val = __shfl_up_sync(0xffffffff, count, i);
    if (id >= i)
      count += val;
  }
  return count;
}

template <typename T>
__global__ void parallel_prefix(const T *atten_mask, int *batch_idx, int *word_idx,
                                const int batch_size, const int max_seq_len) {
  const int tid = threadIdx.x;
  const int warp_count = blockDim.x >> 5;
  int warp_id = tid >> 5;
  int warp_tid = tid & 0x1F;

  extern __shared__ int base[];

  int *seq_len = base;
  int *seq_offset = base + batch_size;

  for (int wid = warp_id; wid < batch_size; wid += warp_count) {
    int count = 0;
    for (int i = warp_tid; i < (max_seq_len + 31) / 32 * 32; i += 32) {
      T mask = i < max_seq_len ? atten_mask[wid * max_seq_len * max_seq_len + i] : (T)0.0f;
      count += __popc(__ballot_sync(0xFFFFFFFF, mask >= (T)0.5f));
    }
    if (warp_tid == 0)
      seq_len[wid] = count;
  }

  __syncthreads();

  if (warp_id == 0) {
    int offset = 0, temp = 0;
    for (int i = warp_tid; i < ((batch_size + 31) / 32) * 32; i += 32) {
      offset = warp_tid == 0 ? temp : 0;
      int len = i < batch_size ? seq_len[i] : 0;
      temp = warpPrefixSum(warp_tid, offset + len);
      if (i < batch_size)
        seq_offset[i] = temp - len;

      temp = __shfl_sync(0xffffffff, temp, 31);
    }
    if (warp_tid == 0)
      seq_offset[batch_size] = temp;
  }

  __syncthreads();

  for (int i = tid; i <= batch_size; i += blockDim.x)
    batch_idx[i] = seq_offset[i];

  for (int wid = warp_id; wid < batch_size; wid += warp_count) {
    int offset = seq_offset[wid];
    for (int i = warp_tid; i < seq_len[wid]; i += 32)
      word_idx[offset + i] = wid * max_seq_len + i;
  }
}

template <typename T>
void build_sequence_length_padding_offset_kernelLauncher(const T *atten_mask, int *batch_idx,
                                                         int *word_idx, int *valid_word_num,
                                                         const int batch_size,
                                                         const int max_seq_len,
                                                         cudaStream_t stream) {
  dim3 block(batch_size * 32);  // one warp per sequence
  if (block.x > 1024)
    block.x = 1024;
  parallel_prefix<<<1, block, (2 * batch_size + 1) * sizeof(int), stream>>>(
      atten_mask, batch_idx, word_idx, batch_size, max_seq_len);
  cudaMemcpyAsync(valid_word_num, batch_idx + batch_size, sizeof(int), cudaMemcpyDeviceToHost,
                  stream);
}

// *********************** compresse transformer input ***********************
template <typename T>
__global__ void compress_bert_input(const T *from_tensor, T *to_tensor, const int *batch_idx,
                                    const int *word_idx, int hidden_dim) {
  int offset = __ldg(&word_idx[blockIdx.x]);
  int dst_idx = blockIdx.x * hidden_dim + threadIdx.x;
  int src_idx = offset * hidden_dim + threadIdx.x;
  ((float4 *)to_tensor)[dst_idx] = ((const float4 *)from_tensor)[src_idx];
}

template <typename T>
void compressBertInput_kernelLauncher(const T *from_tensor, T *to_tensor, int *batch_idx,
                                      int *word_idx, int valid_word_num, int batch_size,
                                      int hidden_dim, cudaStream_t stream) {
  dim3 grid(valid_word_num);
  dim3 block(hidden_dim / 4);  // assert(hidden_dim / 4 <= 1024);
  compress_bert_input<<<grid, block, 0, stream>>>(from_tensor, to_tensor, batch_idx, word_idx,
                                                  hidden_dim / 4);
}

// *********************** add bias input restore transformer output ***********************

template <typename T>
__global__ void add_bias_input(T *out, const T *input, const T *bias, int n);

template <typename T>
__global__ void add_bias_input_restore_output(const T *out, const T *input, const T *bias, int n,
                                              T *out2, const int *batch_idx, const int *word_idx,
                                              const int seq_len);

template <typename T>
__global__ void add_bias_half_input(T *out, const T *input, const T *bias, int n);

template <typename T>
__global__ void add_bias_half_input_restore_output(const T *out, const T *input, const T *bias,
                                                   int n, T *out2, const int *batch_idx,
                                                   const int *word_idx, const int seq_len);
}  // namespace bytetransformer
