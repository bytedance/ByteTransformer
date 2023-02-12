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
#include "gemm.h"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

#include <iostream>
using DataType_ = half;
using namespace std;

#ifndef CUTLASS_CHECK
#define CUTLASS_CHECK(status)                                                        \
  {                                                                                  \
    cutlass::Status error = status;                                                  \
    if (error != cutlass::Status::kSuccess) {                                        \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " \
                << "[" << __FILE__ << ":" << __LINE__ << "]" << std::endl;           \
      exit(EXIT_FAILURE);                                                            \
    }                                                                                \
  }
#endif

namespace cutlass {
namespace epilogue {
namespace thread {

__forceinline__ __device__ float copysignf_pos(float a, float b) {
  return __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
}

// Compute hyperbolic tangent for < sm_75. maxulperr = 108.82848, maxrelerr = 9.3450e-6
__forceinline__ __device__ float __tanhf(float a) {
  const float L2E = 1.442695041f;
  float e, r, s, t, d;
  s = fabsf(a);
  t = -L2E * 2.0f * s;
  asm("ex2.approx.ftz.f32 %0,%1;\n\t" : "=f"(e) : "f"(t));
  d = e + 1.0f;
  asm("rcp.approx.ftz.f32 %0,%1;\n\t" : "=f"(r) : "f"(d));
  r = fmaf(e, -r, r);
  if (s < 4.997253418e-3f)
    r = a;
  if (!__isnan(a))
    r = copysignf_pos(r, a);
  return r;
}

__forceinline__ __device__ float fast_tanh(float lhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  float ret;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(ret) : "f"(lhs));
  return ret;
#else
  // return ::tanhf(lhs);
  return __tanhf(lhs);
#endif
}

// GELUFAST operator
template <typename T>
struct GELUFAST {
  __device__ T operator()(T const &scalar_t) const {
    float scalar = static_cast<float>(scalar_t);
    return T(
        0.5f * scalar *
        (1.0f + fast_tanh(0.7978845608028654f * (scalar + 0.044715f * scalar * scalar * scalar))));
  }
};

template <>
struct GELUFAST<float> {
  __device__ float operator()(float const &scalar) const {
    return 0.5f * scalar *
           (1.0f +
            fast_tanh(0.7978845608028654f * (scalar + 0.044715f * scalar * scalar * scalar)));
  }
};

template <typename T, int N>
struct GELUFAST<Array<T, N>> {
  __device__ Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    GELUFAST<T> gelu_op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(rhs.size()); ++i)
      y[i] = gelu_op(rhs[i]);
    return y;
  }
};

/// Applies a linear combination operator to an array of elements.
/// D = alpha * accumulator + beta * source + uniform
template <typename ElementOutput_,  ///< Data type used to load and store tensors
          int Count,                ///< Number of elements computed per operation
          typename ElementAccumulator_ = ElementOutput_,  ///< Accumulator data type
          typename ElementCompute_ =
              ElementOutput_,  ///< Data type used to compute linear combination
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class LinearCombinationGELUFAST {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {
    ElementCompute alpha;  ///< scales accumulators
    ElementCompute beta;   ///< scales source tensor
    ElementCompute const
        *alpha_ptr;  ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const
        *beta_ptr;  ///< pointer to source scalar - if not null, loads it from memory

    // Methods
    CUTLASS_HOST_DEVICE
    Params()
        : alpha(ElementCompute(1)),
          beta(ElementCompute(0)),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {
    }

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta)
        : alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) {
    }

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr, ElementCompute const *beta_ptr)
        : alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {
    }
  };

 private:
  ElementCompute alpha_, beta_;

 public:
  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  LinearCombinationGELUFAST(Params const &params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    CUTLASS_UNUSED(k_partition_count);
    if (k_partition)
      beta_ = ElementCompute(1);
  }

  /// Computes: D = gelu( alpha * accumulator + beta * source )
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentOutput const &source) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    ComputeFragment converted_source = source_converter(source);

    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    multiplies<ComputeFragment> mul_add_source;
    ComputeFragment intermediate =
        mul_add_source(beta_, converted_source);  // X = beta * C + uniform

    multiply_add<ComputeFragment> mul_add_accumulator;
    intermediate =
        mul_add_accumulator(alpha_, converted_accumulator, intermediate);  // D = alpha * Accum + X

    GELUFAST<ComputeFragment> gelu;
    intermediate = gelu(intermediate);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
    return destination_converter(intermediate);
  }

  /// Computes: D = gelu( alpha * accumulator )
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    multiplies<ComputeFragment> mul_add_accumulator;
    ComputeFragment intermediate =
        mul_add_accumulator(alpha_, converted_accumulator);  // D = alpha * Accum

    GELUFAST<ComputeFragment> gelu;
    intermediate = gelu(intermediate);

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
    return destination_converter(intermediate);
  }
};

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

namespace bytetransformer {
using ElementAccumulator = cutlass::half_t;         // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D
using LayoutInputA = cutlass::layout::RowMajor;     // m*k row-major == k*m col-major
using LayoutInputB = cutlass::layout::RowMajor;     // k*n row-major == n*k col-major
using LayoutOutput = cutlass::layout::RowMajor;     // m*n row-major == n*m col-major
using MMAOp = cutlass::arch::OpClassTensorOp;
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGELUFAST<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width
                                                       // of math instructions in the epilogue too.
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

#define _block_m 128
#define _block_n 128
#define _block_k 32
#define _warp_m 64
#define _warp_n 64
#define _warp_k 32

#define GEMM_TYPE(BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, INST_M, INST_N, INST_K,                       \
                  NUM_STAGES)                                                                                      \
  using ShapeMMAThreadBlock_##BLOCK_M##_##BLOCK_N##_##BLOCK_K =                                                    \
      cutlass::gemm::GemmShape<BLOCK_M, BLOCK_N, BLOCK_K>;                                                         \
  using ShapeMMAWarp_##WARP_M##_##WARP_N##_##WARP_K =                                                              \
      cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>;                                                            \
  using ShapeMMAOp_##INST_M##_##INST_N##_##INST_K =                                                                \
      cutlass::gemm::GemmShape<INST_M, INST_N, INST_K>;                                                            \
  using Gemm_##BLOCK_M##_##BLOCK_N##_##BLOCK_K##_##WARP_M##_##WARP_N##_##WARP_K##_##INST_M##_##INST_N##_##INST_K = \
      cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,                        \
                                  ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch,                  \
                                  ShapeMMAThreadBlock_##BLOCK_M##_##BLOCK_N##_##BLOCK_K,                           \
                                  ShapeMMAWarp_##WARP_M##_##WARP_N##_##WARP_K,                                     \
                                  ShapeMMAOp_##INST_M##_##INST_N##_##INST_K, EpilogueOp,                           \
                                  SwizzleThreadBlock, NUM_STAGES>;

#define GEMM_BIAS_GELU(BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, INST_M, INST_N, INST_K) \
  Gemm_##BLOCK_M##_##BLOCK_N##_##BLOCK_K##_##WARP_M##_##WARP_N##_##WARP_K##_##INST_M##_##INST_N##_##INST_K

#define GEMM_INIT(BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, INST_M, INST_N, INST_K)                          \
  typename Gemm_##BLOCK_M##_##BLOCK_N##_##BLOCK_K##_##WARP_M##_##WARP_N##_##WARP_K##_##INST_M##_##INST_N##_##INST_K:: \
      Arguments args{problem_size,                                                                                    \
                     {(ElementInputB *)A_, k_},                                                                       \
                     {(ElementInputA *)B_, n_},                                                                       \
                     {(ElementOutput *)bias_, 0},                                                                     \
                     {(ElementOutput *)C_, n_},                                                                       \
                     {alpha, beta},                                                                                   \
                     split_k_slices_};

__inline__ __device__ float tanht(float lhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  float ret;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(ret) : "f"(lhs));
  return ret;
#else
  return ::tanhf(lhs);
#endif
}

template <typename T>
__inline__ __device__ T gelu(T x) {
#ifdef CUDA11_MODE
  float cdf = 0.5f * (1.0f + tanht((0.7978845608028654f * (x + 0.044715f * x * x * x))));
#else
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
#endif
  return x * cdf;
}

template <>
__inline__ __device__ half2 gelu(half2 val) {
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp = __half22float2(val);

#ifdef CUDA11_MODE
  tmp.x = 0.5f * (1.0f + tanht((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y = 0.5f * (1.0f + tanht((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
#else
  tmp.x = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
#endif
  return __hmul2(val, __float22half2_rn(tmp));
}

template <ActType act, typename T>
__global__ void add_bias_act(T *output, const T *bias, const int M, const int N) {
  int row_offset = blockIdx.x * N;
  for (int tid = threadIdx.x; tid < N; tid += blockDim.x) {
    T out = output[row_offset + tid] + __ldg(&bias[tid]);
    out = act_fun<act>(out);
    output[row_offset + tid] = out;
  }
}

template <typename T>
__global__ void add_bias_gelu(T *output, const T *bias, const int M, const int N);

template <typename T>
void gemm_bias_gelu(const T *A_, const T *B_, T *C_, const T *bias_, int m_, int k_, int n_,
                    cudaStream_t stream, cublasHandle_t cublas_handle, int cublasAlgo, int arch);

template <typename T>
void gemm_bias_relu(const T *A_, const T *B_, T *C_, const T *bias_, int m_, int k_, int n_,
                    cudaStream_t stream, cublasHandle_t cublas_handle, int cublasAlgo, int arch);
}  // namespace bytetransformer
