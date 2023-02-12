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

namespace bytetransformer {
#define FINAL_MASK 0xffffffff

__device__ __forceinline__ int getLaneId() {
  int laneId;
  asm("mov.s32 %0, %laneid;" : "=r"(laneId));
  return laneId;
}

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  return wid == 0 ? warpReduceSum(threadIdx.x < (blockDim.x >> 5) ? shared[lane] : (T)0.0f) : 0.0f;
}

__inline__ __device__ __half2 warpReduceSum(__half2 val) {
  half2 tmp_val;
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp_val = __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    val = __hadd2(tmp_val, val);
  }
  return val;
}

__inline__ __device__ __half __half2add(__half2 val) {
  return __hadd(val.x, val.y);
}

__inline__ __device__ __half blockReduceSum(__half2 val) {
  static __shared__ __half2 shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<__half2>(val);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  return (__half)(wid == 0 ? warpReduceSum(threadIdx.x < (blockDim.x >> 5)
                                               ? (float)__half2add(shared[lane])
                                               : 0.0f)
                           : 0.0f);
}

template <typename T>
__inline__ __device__ T max_(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max_(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax(val);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  return wid == 0 ? warpReduceMax(threadIdx.x < (blockDim.x >> 5) ? shared[lane] : (T)-1e20f)
                  : (T)-1e20f;
}
}  // namespace bytetransformer
