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
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace bytetransformer {
enum class OperationType { FP32, HALF };

template <OperationType OpType>
class Traits;

template <>
class Traits<OperationType::FP32> {
 public:
  typedef float DataType;
  // cuBLAS parameters
  static cudaDataType_t const computeType = CUDA_R_32F;
  static cudaDataType_t const AType = CUDA_R_32F;
  static cudaDataType_t const BType = CUDA_R_32F;
  static cudaDataType_t const CType = CUDA_R_32F;
  static const int algo = -1;
};

template <>
class Traits<OperationType::HALF> {
 public:
  typedef __half DataType;
  // cuBLAS parameters
  static cudaDataType_t const computeType = CUDA_R_16F;
  static cudaDataType_t const AType = CUDA_R_16F;
  static cudaDataType_t const BType = CUDA_R_16F;
  static cudaDataType_t const CType = CUDA_R_16F;
  static const int algo = 99;
};

typedef struct {
  int *batch_idx;
  int *word_idx;
  int valid_word_num;
} ET_Param;

enum ModelType { Bert };

enum ActType { Relu, Sigmoid, SoftPlus, No };

template <ActType act, typename T>
__inline__ __device__ T act_fun(T val) {
  if (act == ActType::Relu)
    return (val <= (T)0.0f) ? (T)0.0f : val;
  else if (act == ActType::SoftPlus)
    return __logf(__expf((float)val) + 1.0f);
  else if (act == ActType::Sigmoid)
    return 1.0f / (1.0f + __expf(-1.0f * (float)val));
  else
    return val;
}

typedef union half4 {
  float2 x;
  half2 h[2];
} half4;

#define PRINT_FUNC_NAME_()                                          \
  do {                                                              \
    std::cout << "[BT][CALL] " << __FUNCTION__ << " " << std::endl; \
  } while (0)

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorString(error);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result)
    throw std::runtime_error(std::string("[BT][ERROR] CUDA runtime error: ") +
                             (_cudaGetErrorEnum(result)) + " " + file + ":" +
                             std::to_string(line) + " \n");
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    ::cutlass::Status error = status;                                                            \
    if (error != ::cutlass::Status::kSuccess) {                                                  \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

}  // namespace bytetransformer
