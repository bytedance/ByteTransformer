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
#include <vector>
#include <iostream>
#include <cuda_fp16.h>
#include <nvToolsExt.h>

#include "c10/cuda/CUDAGuard.h"
#include "ATen/cuda/CUDAContext.h"
#include "torch/all.h"

#define CHECK_TYPE(x, st) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type: " #x)
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st) \
  CHECK_CUDA(x);           \
  CHECK_CONTIGUOUS(x);     \
  CHECK_TYPE(x, st)
#define CHECK_INPUT_LOOSE(x) \
  CHECK_CUDA(x);             \
  CHECK_CONTIGUOUS(x)
#define CHECK_SEQ_LEN(x) TORCH_CHECK(x <= 1024, "Input seq_len too long")
#define PRINT_TENSOR(x) std::cout << #x << ":\n" << x << std::endl
#define PRINT_TENSOR_SIZE(x) std::cout << "size of " << #x << ": " << x.sizes() << std::endl
#define CHECK_FROM_SEQ_LEN(x) TORCH_CHECK(x <= 32, "Query from_seq_len too long")
#define CHECK_TO_SEQ_LEN(x) TORCH_CHECK(x <= 32, "Key to_seq_len too long")
#define CHECK_HEAD_SIZE(x) TORCH_CHECK(x == 16, "Only support head_size = 16")

namespace bytetransformer {
namespace torch_ext {
template <typename T>
inline T *get_ptr(torch::Tensor &t) {
  return reinterpret_cast<T *>(t.data_ptr());
}

template <typename T>
inline const T *get_ptr(const torch::Tensor &t) {
  return reinterpret_cast<T *>(t.data_ptr());
}
}  // namespace torch_ext
}  // namespace bytetransformer
