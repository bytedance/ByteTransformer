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
void dense_layer_kernel_launcher(const float *in, const float *weight, float *out, const int M,
                                 const int K, const int N, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo = -1);

void dense_layer_kernel_launcher(const __half *in, const __half *weight, __half *out, const int M,
                                 const int K, const int N, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo = 99);

void cublas_Gemm_Strided_Batched(const float *A, const float *B, float *out, const int M,
                                 const int K, const int N, const int batch_count,
                                 cublasOperation_t trans_A, cublasOperation_t trans_B, float alpha,
                                 float beta, cublasHandle_t cublas_handle, cudaStream_t stream,
                                 int cublasAlgo = -1);

void cublas_Gemm_Strided_Batched(const __half *A, const __half *B, __half *out, const int M,
                                 const int K, const int N, const int batch_count,
                                 cublasOperation_t trans_A, cublasOperation_t trans_B,
                                 __half alpha, __half beta, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo = 99);
}  // namespace bytetransformer
