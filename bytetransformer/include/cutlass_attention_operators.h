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
#include "cutlass/arch/memory.h"
#include "cutlass/contrib/args_pack_def.h"
#include "cutlass/contrib/functional.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

namespace bytetransformer {
namespace cutlass_ops {

/// GemmGropued arguments helper operators
template <int SizePerHead, int Alignment, bool IsGemm0>
class AttentionGemmProblemSize {
 public:
  struct Params {
    int const *seqlen_offsets = nullptr;
    int seqlen = 0;
  };

 private:
  int const *seqlen_offsets_;
  int const seqlen_;

 public:
  CUTLASS_HOST_DEVICE
  AttentionGemmProblemSize() : seqlen_offsets_(nullptr), seqlen_(0) {
  }
  CUTLASS_HOST_DEVICE
  AttentionGemmProblemSize(Params const &params)
      : seqlen_offsets_(params.seqlen_offsets), seqlen_(params.seqlen) {
  }
  CUTLASS_DEVICE
  cutlass::gemm::GemmCoord operator()(int problem_idx) const {
    int valid_seqlen = seqlen_offsets_ == nullptr ? seqlen_
                                                  : __ldg(&seqlen_offsets_[problem_idx + 1]) -
                                                        __ldg(&seqlen_offsets_[problem_idx]);
    int valid_seqlen_pad = (valid_seqlen + Alignment - 1) / Alignment * Alignment;
    if constexpr (std::bool_constant<IsGemm0>::value) {
      return {valid_seqlen, valid_seqlen_pad, SizePerHead};
    }
    return {valid_seqlen, SizePerHead, valid_seqlen_pad};
  }
};

class AttentionBatchCountGeneratorOp {
 public:
  struct Params {
    int head_num;
  };

 private:
  int const head_num_;

 public:
  CUTLASS_HOST_DEVICE
  AttentionBatchCountGeneratorOp() : head_num_(0) {
  }

  CUTLASS_HOST_DEVICE
  AttentionBatchCountGeneratorOp(Params const &params) : head_num_(params.head_num) {
  }
  CUTLASS_DEVICE
  int operator()(int problem_idx) const {
    return head_num_;
  }
};

template <typename Element_>
class AttentionTensorParamGeneratorOp {
 public:
  using Element = Element_;
  struct Params {
    Element *ptr;
    int ldm;
    int64_t batch_stride;
    int64_t problem_stride;
    int const *seqlen_offsets = nullptr;
  };

 private:
  Element *ptr_;
  int const ldm_;
  int64_t const batch_stride_;
  int64_t const problem_stride_;
  int const *seqlen_offsets_;

 public:
  CUTLASS_HOST_DEVICE
  AttentionTensorParamGeneratorOp()
      : ptr_(nullptr), ldm_(0), batch_stride_(0), problem_stride_(0), seqlen_offsets_(nullptr) {
  }

  CUTLASS_HOST_DEVICE
  AttentionTensorParamGeneratorOp(Params const &params)
      : ptr_(params.ptr),
        ldm_(params.ldm),
        batch_stride_(params.batch_stride),
        problem_stride_(params.problem_stride),
        seqlen_offsets_(params.seqlen_offsets) {
  }
  CUTLASS_DEVICE
  cutlass::contrib::TensorParams<Element> operator()(int problem_idx, int batch_idx) const {
    cutlass::contrib::TensorParams<Element> ret;
    ret.ldm = ldm_;
    if (seqlen_offsets_ == nullptr) {
      // no remove padding
      ret.ptr = ptr_ + problem_idx * problem_stride_ + batch_idx * batch_stride_;
    } else {
      // remove padding
      int batch_offset = __ldg(&seqlen_offsets_[problem_idx]);
      ret.ptr = ptr_ + batch_offset * problem_stride_ + batch_idx * batch_stride_;
    }
    return ret;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Custom Prologue/Epilogue Operators
////////////////////////////////////////////////////////////////////////////////

// op_type sum, max, min, etc
template <typename T, int N, typename op_type>
struct ReductionThreadScope {
  CUTLASS_DEVICE
  cutlass::Array<T, 1> operator()(cutlass::Array<T, N> const &rhs) const {
    cutlass::Array<T, 1> y;
    op_type op;

    y[0] = rhs[0];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < N; ++i) {
      y[0] = op(y[0], rhs[i]);
    }

    return y;
  }
};

template <typename T, typename op_type, int AccessWidth>
struct ReductionWarpScope {
  static constexpr int kAccessWidth = AccessWidth;
  static_assert((kAccessWidth & (kAccessWidth - 1)) == 0, "AccessWidth must be power of 2");

  CUTLASS_DEVICE
  T operator()(T val) const {
    T res = val;
    op_type op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < kAccessWidth; i <<= 1) {
      T other = __shfl_xor_sync(0xFFFFFFFF, res, i, kAccessWidth);
      res = op(res, other);
    }

    return res;
  }
};

template <typename T, int N, typename op_type, int AccessWidth>
struct ReductionWarpScope<cutlass::Array<T, N>, op_type, AccessWidth> {
  CUTLASS_DEVICE
  cutlass::Array<T, N> operator()(cutlass::Array<T, N> const &rhs) const {
    cutlass::Array<T, N> y;
    ReductionWarpScope<T, op_type, AccessWidth> warp_reduction_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(rhs.size()); ++i) {
      y[i] = warp_reduction_op(rhs[i]);
    }

    return y;
  }
};

template <typename ComputeFragment, typename op_type, int AccessWidth>
struct ReductionFragmentinTensorCoreShape {
  using ElementCompute = typename ComputeFragment::Element;
  using FragmentReductionAccumulator = cutlass::Array<ElementCompute, 1>;

  CUTLASS_DEVICE
  FragmentReductionAccumulator operator()(ComputeFragment const &rhs) const {
    FragmentReductionAccumulator thread_red_res;
    FragmentReductionAccumulator res;
    ReductionThreadScope<ElementCompute, ComputeFragment::kElements, op_type> reduction_in_thread;
    ReductionWarpScope<FragmentReductionAccumulator, op_type, AccessWidth> reduction_in_warp;

    thread_red_res = reduction_in_thread(rhs);
    res = reduction_in_warp(thread_red_res);

    return res;
  }
};

/// Partial reduce max and exp(sum - max) of each threadblock
template <typename ElementOutput_, int Count, typename ElementAccumulator_,
          typename ElementCompute_,
          cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
class SoftmaxPartialEpilogueOp {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static constexpr const int kCount = Count;
  static constexpr bool kOperatorNeedCoord = true;

  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

  using FragmentReductionAccumulator = cutlass::Array<ElementCompute, 1>;

  static cutlass::FloatRoundStyle const kRound = Round;

 public:
  struct Params {
    ElementCompute scalar;
    ElementCompute beta;
    int head_num;
    int row_num;
    int threadblock_num_in_row;
    ElementCompute *ptr;
  };

 private:
  const ElementCompute scalar_;
  const ElementCompute beta_;
  const int head_num_;
  const int row_num_;
  const int threadblock_num_in_row_;
  const int problem_idx_;
  ElementCompute *ptr_;

 public:
  CUTLASS_HOST_DEVICE
  SoftmaxPartialEpilogueOp(Params const &params, int problem_idx, int batch_idx)
      : scalar_(params.scalar),
        beta_(params.beta),
        head_num_(params.head_num),
        row_num_(params.row_num),
        threadblock_num_in_row_(params.threadblock_num_in_row),
        problem_idx_(problem_idx) {
    ptr_ = params.ptr +
           (problem_idx * head_num_ + batch_idx) * row_num_ * threadblock_num_in_row_ * 2;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ != ElementCompute(0);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    FragmentOutput ret;
    ret.clear();
    return ret;
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator,
                            FragmentOutput const &source) const {
    FragmentOutput ret;
    ret.clear();
    return ret;
  }

  template <typename Iterator, typename SourceFragment>
  CUTLASS_HOST_DEVICE typename Iterator::Fragment operator()(
      typename Iterator::Fragment const &frag, SourceFragment const &source,
      Iterator const &iterator) const {
    using ThreadMap = typename Iterator::ThreadMap;
    static_assert(ThreadMap::kElementsPerAccess == kCount,
                  "kElementsPerAccess must match thread map");
    static_assert(Iterator::Fragment::kElements == SourceFragment::kElements,
                  "kElements of source must match");
    static_assert(std::is_same<typename Iterator::Fragment::Element, ElementAccumulator>::value,
                  "accumulator element must match");
    static_assert(ThreadMap::Iterations::kCluster == 1, "Iteration of cluster must be 1");
    static_assert(ThreadMap::Iterations::kGroup == 1, "Iteration of group must be 1");
    static_assert(std::is_same<ElementCompute, float>::value, "ElementCompute must be float");
    // static_assert(ThreadMap::Iterations::kColumn == 1, "Iteration of column must be 1");

    cutlass::MatrixCoord thread_offset = iterator.thread_offset();

    const int threadblock_column_idx =
        thread_offset.column() / (ThreadMap::Shape::kColumn * ThreadMap::Count::kColumn);

    const int kCountRow = ThreadMap::Iterations::kColumn * kCount;
    const int kAccessWidth = ThreadMap::Detail::kAccessWidth;
    using FragmentRow = cutlass::Array<ElementAccumulator, kCountRow>;
    using ComputeFragmentRow = cutlass::Array<ElementCompute, kCountRow>;

    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCountRow, Round>
        accumulator_converter;
    cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCountRow, Round>
        source_converter;
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCountRow, Round>
        output_converter;

    typename Iterator::Fragment ret;
    FragmentRow *ret_ptr = reinterpret_cast<FragmentRow *>(&ret);
    FragmentRow const *frag_ptr = reinterpret_cast<FragmentRow const *>(&frag);
    FragmentRow const *source_ptr = reinterpret_cast<FragmentRow const *>(&source);

    const bool is_write_thread = !(threadIdx.x % kAccessWidth);

    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
      const int row_idx = thread_offset.row() + row * ThreadMap::Delta::kRow;
      FragmentRow const &frag_row = frag_ptr[row];
      FragmentRow source_row = source_ptr[row];

      cutlass::contrib::multiply_add<FragmentRow> mask_converter;
      source_row = mask_converter(source_row, ElementOutput(10000.0), ElementOutput(-10000));

      ComputeFragmentRow converted_source = source_converter(source_row);
      ComputeFragmentRow converted_frag = accumulator_converter(frag_row);
      cutlass::multiply_add<ComputeFragmentRow> mul_add_op;

      // add mask
      ComputeFragmentRow intermediate = mul_add_op(scalar_, converted_frag, converted_source);
      FragmentRow output_frag = output_converter(intermediate);
      ret_ptr[row] = output_frag;

      ReductionFragmentinTensorCoreShape<ComputeFragmentRow, cutlass::maximum<ElementCompute>,
                                         kAccessWidth>
          max_op;

      FragmentReductionAccumulator max_val = max_op(intermediate);

      cutlass::minus<ComputeFragmentRow> minus_op;
      intermediate = minus_op(intermediate, max_val.at(0));

      cutlass::fast_exp_op<ComputeFragmentRow> exp_op;
      intermediate = exp_op(intermediate);

      ReductionFragmentinTensorCoreShape<ComputeFragmentRow, cutlass::plus<ElementCompute>,
                                         kAccessWidth>
          sum_op;
      FragmentReductionAccumulator sum = sum_op(intermediate);

      ElementCompute *row_ptr = ptr_ + row_idx * threadblock_num_in_row_ * 2;
      using StoreType = cutlass::AlignedArray<ElementCompute, 2>;
      StoreType value;
      value[0] = max_val.at(0);
      value[1] = sum.at(0);

      cutlass::arch::global_store<StoreType, sizeof(StoreType)>(
          value, reinterpret_cast<void *>(row_ptr + threadblock_column_idx * 2),
          row_idx < row_num_ && is_write_thread);
    }
    return ret;
  }
};

template <typename T, int ThreadblockN>
__global__ void softmax_reduction_kernel(const T *buf, const int *seqlen_offsets, T *out,
                                         const int column, const bool is_remove_padding) {
  const int head_num = gridDim.y;
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int row_num = blockDim.x;
  const int row_idx = threadIdx.x;
  const int valid_seqlen =
      is_remove_padding
          ? (__ldg(&seqlen_offsets[batch_idx + 1]) - __ldg(&seqlen_offsets[batch_idx]))
          : row_num;
  T *out_buf = out + ((batch_idx * head_num + head_idx) * row_num + row_idx) * 2;
  T max_val(0.0f);
  T sum_val(0.0f);

  if (row_idx < valid_seqlen) {
    const int valid_column = (valid_seqlen + ThreadblockN - 1) / ThreadblockN;

    const T *row_buf = buf + ((batch_idx * head_num + head_idx) * row_num + row_idx) * column * 2;
    max_val = __ldg(row_buf);
    for (int i = 1; i < valid_column; ++i) {
      max_val = max(max_val, __ldg(row_buf + i * 2));
    }

    for (int i = 0; i < valid_column; ++i) {
      sum_val += __ldg(row_buf + i * 2 + 1) * expf(__ldg(row_buf + i * 2) - max_val);
    }
    // printf("%d %d %d %f %f\n", batch_idx, row_idx, valid_column, max_val, sum_val);
  }
  out_buf[0] = max_val;
  out_buf[1] = sum_val;
}

template <typename T, int ThreadblockN>
void softmax_reduction_kernel_launcher(const T *buf, const int *seqlen_offsets, T *out,
                                       const int batch_size, const int head_num, const int seq_len,
                                       const int column, const bool is_remove_padding,
                                       cudaStream_t stream) {
  dim3 grid(batch_size, head_num);
  dim3 block(seq_len);
  assert(seq_len <= 1024);
  softmax_reduction_kernel<T, ThreadblockN>
      <<<grid, block, 0, stream>>>(buf, seqlen_offsets, out, column, is_remove_padding);
}

/// Custom prologue op for softmax
template <typename Element_, typename ElementCompute_,
          cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
class SoftmaxPartialPrologueOp {
 public:
  using Element = Element_;
  using ElementCompute = ElementCompute_;

  static_assert(std::is_same<Element, cutlass::half_t>::value, "Only support half");
  static_assert(std::is_same<ElementCompute, float>::value, "Only support float for compute");
  static constexpr cutlass::FloatRoundStyle kRound = Round;

  template <typename WarpLoadIterator>
  struct Detail {
    using WarpShape = typename WarpLoadIterator::Shape;
    using InstructionShape = typename WarpLoadIterator::InstructionShape;
    static_assert(WarpLoadIterator::kOperand == cutlass::gemm::Operand::kA,
                  "SoftmaxPartialPrologue only supports A operand");
    static_assert(InstructionShape::kRow == 16 && InstructionShape::kColumn == 8,
                  "Only support 16x8 Instruction Shape");

    static constexpr int kRows = WarpShape::kRow / InstructionShape::kRow * 2;
    static constexpr int kRowDelta = 8;
    using Fragment = cutlass::Array<ElementCompute, 2 * kRows>;

    static constexpr int calc_thread_row_offset(int lane_id) {
      return lane_id >> 2;
    }
  };

 public:
  struct Params {
    int head_num;
    int row_num;
    ElementCompute const *ptr;
  };

 private:
  const int head_num_;
  const int row_num_;
  ElementCompute const *ptr_;
  const int warp_row_;
  const int column_extent_;
  bool is_residual_tile_;
  int kgroup_index_;

 public:
  CUTLASS_HOST_DEVICE
  SoftmaxPartialPrologueOp(Params const &params, int problem_idx, int batch_idx,
                           cutlass::MatrixCoord extent, cutlass::MatrixCoord warp_offset)
      : head_num_(params.head_num),
        row_num_(params.row_num),
        warp_row_(warp_offset.row()),
        column_extent_(extent.column()),
        is_residual_tile_(true),
        kgroup_index_(0) {
    ptr_ = params.ptr + (problem_idx * head_num_ + batch_idx) * row_num_ * 2;
  }

  template <typename WarpLoadIterator>
  CUTLASS_DEVICE void load(typename Detail<WarpLoadIterator>::Fragment &frag) {
    if (!is_residual_tile_) {
      return;
    }
    using IteratorDetail = Detail<WarpLoadIterator>;
    using Fragment = typename IteratorDetail::Fragment;
    int lane_id = threadIdx.x % 32;
    const int initial_row = warp_row_ + IteratorDetail::calc_thread_row_offset(lane_id);
    using AccessType = cutlass::Array<ElementCompute, 2>;
    using LoadType = cutlass::AlignedArray<ElementCompute, 2>;
    AccessType *access_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < IteratorDetail::kRows; ++i) {
      // ith-row
      int row_idx = initial_row + i * IteratorDetail::kRowDelta;
      LoadType value;
      cutlass::arch::global_load<LoadType, sizeof(LoadType)>(
          value, (void const *)(ptr_ + row_idx * 2), row_idx < row_num_);
      access_ptr[i] = *reinterpret_cast<AccessType *>(&value);
    }
  }

  template <typename WarpLoadIterator>
  CUTLASS_DEVICE void operator++() {
    is_residual_tile_ = false;
  }

  template <typename WarpLoadIterator>
  CUTLASS_DEVICE void set_kgroup_index(int k_group) {
    kgroup_index_ = k_group;
  }

  template <typename WarpLoadIterator>
  CUTLASS_HOST_DEVICE typename WarpLoadIterator::Fragment operator()(
      typename WarpLoadIterator::Fragment const &mma_frag,
      typename Detail<WarpLoadIterator>::Fragment const &frag) const {
    using IteratorDetail = Detail<WarpLoadIterator>;
    using FragmentMma = typename WarpLoadIterator::Fragment;
    using Fragment = typename IteratorDetail::Fragment;

    constexpr int kAccessSize = 2;
    static_assert(FragmentMma::kElements == kAccessSize * IteratorDetail::kRows,
                  "FragmentMma size not valid");
    static_assert(std::is_same<Element, typename FragmentMma::Element>::value, "Element mismatch");
    using AccessType = cutlass::Array<Element, kAccessSize>;
    using ComputeFragment = cutlass::Array<ElementCompute, kAccessSize>;

    cutlass::NumericArrayConverter<ElementCompute, Element, kAccessSize, Round> compute_converter;
    cutlass::NumericArrayConverter<Element, ElementCompute, kAccessSize, Round> output_converter;

    FragmentMma ret;
    const int residual = column_extent_ % IteratorDetail::WarpShape::kColumn;
    // set residual values to zero
    if (is_residual_tile_ && residual &&
        kgroup_index_ * IteratorDetail::InstructionShape::kColumn >= residual) {
      ret.clear();
      return ret;
    }
    AccessType const *mma_frag_ptr = reinterpret_cast<AccessType const *>(&mma_frag);
    AccessType *ret_ptr = reinterpret_cast<AccessType *>(&ret);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < IteratorDetail::kRows; ++i) {
      ElementCompute max_row = frag[i * 2];
      ElementCompute sum_row = frag[i * 2 + 1] + ElementCompute(1e-6);

      ComputeFragment converted_frag = compute_converter(mma_frag_ptr[i]);
      cutlass::minus<ComputeFragment> minus_op;
      ComputeFragment intermediate = minus_op(converted_frag, max_row);
      cutlass::fast_exp_op<ComputeFragment> exp_op;
      intermediate = exp_op(intermediate);
      cutlass::divides<ComputeFragment> div_op;
      intermediate = div_op(intermediate, sum_row);
      ret_ptr[i] = output_converter(intermediate);
    }
    return ret;
  }
};

template <typename ElementOutput_,  ///< Data type used to load and store tensors
          int Count,                ///< Number of elements computed per operation
          typename ElementAccumulator_ = ElementOutput_,  ///< Accumulator data type
          typename ElementCompute_ = ElementOutput_,      ///< Data type used to compute
          cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
class SoftmaxFusedEpilogue {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = float;  // ElementCompute_;

  static int const kCount = Count;
  // hard coded access witdh, for turing and ampere
  static constexpr int kAccessWidth = 4;

  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

  using FragmentReductionAccumulator = cutlass::Array<ElementCompute, 1>;

  static cutlass::FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {
    ElementCompute scalar;  ///< scales accumulators
    ElementCompute beta;    ///< scales source tensor

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() : scalar(ElementCompute(1)), beta(ElementCompute(0)) {
    }

    CUTLASS_HOST_DEVICE
    Params(ElementCompute scalar, ElementCompute beta) : scalar(scalar), beta(beta) {
    }
  };

 private:
  //
  // Data members
  //

  ElementCompute scalar_;
  ElementCompute beta_;

 public:
  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  SoftmaxFusedEpilogue(Params const &params) {
    beta_ = params.beta;
    scalar_ = params.scalar;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  /// Computes linear scaling: D = scalar * accumulator + beta * mask_val
  CUTLASS_HOST_DEVICE
  FragmentAccumulator operator()(FragmentAccumulator const &accumulator,
                                 FragmentOutput const &source) const {
    // Convert source to interal compute numeric type
    cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    cutlass::contrib::multiply_add<FragmentOutput> mask_converter;
    FragmentOutput source_mask =
        mask_converter(source, ElementOutput(10000.0), ElementOutput(-10000.0));

    ComputeFragment converted_source = source_converter(source_mask);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations

    ComputeFragment intermediate;

    // multiplies<ComputeFragment> mul_add_source;
    cutlass::contrib::multiply_add<ComputeFragment> mul_add_accumulator;

    intermediate = mul_add_accumulator(scalar_, converted_accumulator, converted_source);

    ReductionFragmentinTensorCoreShape<ComputeFragment, cutlass::maximum<ElementCompute>,
                                       kAccessWidth>
        max_op;
    FragmentReductionAccumulator max_val = max_op(intermediate);

    cutlass::minus<ComputeFragment> minus_op;
    intermediate = minus_op(intermediate, max_val.at(0));
    cutlass::fast_exp_op<ComputeFragment> exp_op;
    intermediate = exp_op(intermediate);

    ReductionFragmentinTensorCoreShape<ComputeFragment, cutlass::plus<ElementCompute>,
                                       kAccessWidth>
        sum_op;
    FragmentReductionAccumulator sum = sum_op(intermediate);
    cutlass::plus<FragmentReductionAccumulator> plus_op;
    sum = plus_op(sum, ElementCompute(1e-6f));

    cutlass::divides<ComputeFragment> div_op;
    ComputeFragment res = div_op(intermediate, sum.at(0));

    // Convert to destination numeric type
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;
    return destination_converter(res);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    // Convert source to interal compute numeric type
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;
    // NumericArrayConverter<float, float, kCount, Round> accumulator_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations

    // ComputeFragment mask_val, intermediate;
    ComputeFragment intermediate;

    cutlass::multiplies<ComputeFragment> mul_add_source;

    intermediate = mul_add_source(scalar_, converted_accumulator);

    ReductionFragmentinTensorCoreShape<ComputeFragment, cutlass::maximum<ElementCompute>,
                                       kAccessWidth>
        max_op;
    FragmentReductionAccumulator max_val = max_op(intermediate);

    cutlass::minus<ComputeFragment> minus_op;
    intermediate = minus_op(intermediate, max_val.at(0));
    cutlass::fast_exp_op<ComputeFragment> exp_op;
    intermediate = exp_op(intermediate);

    ReductionFragmentinTensorCoreShape<ComputeFragment, cutlass::plus<ElementCompute>,
                                       kAccessWidth>
        sum_op;
    FragmentReductionAccumulator sum = sum_op(intermediate);
    cutlass::plus<FragmentReductionAccumulator> plus_op;
    sum = plus_op(sum, ElementCompute(1e-6f));

    cutlass::divides<ComputeFragment> div_op;
    ComputeFragment res = div_op(intermediate, sum.at(0));

    // Convert to destination numeric type
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    return destination_converter(res);
  }
};

/// Add Bias to Query or Key
template <typename Element_, int SizePerHead, bool Roformer = false>
class QKBiasAddOp {
 public:
  using Element = Element_;
  static constexpr int kSizePerHead = SizePerHead;
  static constexpr bool kRoformer = Roformer;

 public:
  template <typename WarpLoadIterator>
  struct Detail {
    using WarpShape = typename WarpLoadIterator::Shape;
    using InstructionShape = typename WarpLoadIterator::InstructionShape;
    // else operand B
    static constexpr bool kIsOperandA = WarpLoadIterator::kOperand == cutlass::gemm::Operand::kA;
    static_assert(InstructionShape::kColumn == 8, "Only support 8 Instruction Column");

    using WarpPitchLinearShape = typename std::conditional<
        kIsOperandA, cutlass::PitchLinearShape<WarpShape::kColumn, WarpShape::kRow>,
        cutlass::PitchLinearShape<WarpShape::kRow, WarpShape::kColumn>>::type;

    static_assert(!(kSizePerHead % WarpPitchLinearShape::kContiguous),
                  "kSizePerHead must be disible by the warp tile contiguous dim");

    using InstructionPitchLinearShape = typename std::conditional<
        kIsOperandA, cutlass::PitchLinearShape<InstructionShape::kColumn, InstructionShape::kRow>,
        cutlass::PitchLinearShape<InstructionShape::kRow, InstructionShape::kColumn>>::type;

    // hard coded for tensor op
    static constexpr int kAccessSize = 2;
    static constexpr int kStridedDelta = 8;
    static constexpr int kContiguousDelta = 8;

    static constexpr int kStridedCount = WarpPitchLinearShape::kStrided / kStridedDelta;
    static constexpr int kContiguousCount = WarpPitchLinearShape::kContiguous / kContiguousDelta;

    // using Fragment = cutlass::Array<Element, kAccessSize * kStridedCount>;
    using Fragment = cutlass::Array<Element, kContiguousCount * kAccessSize>;

    static constexpr int calc_thread_strided_offset(int lane_id) {
      return lane_id >> 2;
    }

    static constexpr int calc_thread_contiguous_offset(int lane_id) {
      return (lane_id & 0x3) * kAccessSize;
    }
  };

  struct Params {
    Element const *ptr;
    int batch_stride;
    int seqlen = 0;
    int const *seqlen_offsets = nullptr;
  };

 private:
  Element const *ptr_;
  int const valid_seqlen_;
  cutlass::MatrixCoord warp_offsets_;

  int kgroup_index_warp_;  // equals to threadblock k dim
  int kgroup_index_;

  // for roformer
  float position_denominator_;

 public:
  CUTLASS_HOST_DEVICE QKBiasAddOp(Params const &params, int problem_idx, int batch_idx,
                                  cutlass::MatrixCoord extent, cutlass::MatrixCoord warp_offsets)
      : ptr_(params.ptr + batch_idx * params.batch_stride),
        valid_seqlen_(params.seqlen_offsets == nullptr
                          ? params.seqlen
                          : __ldg(&params.seqlen_offsets[problem_idx + 1]) -
                                __ldg(&params.seqlen_offsets[problem_idx])),
        warp_offsets_(warp_offsets),
        kgroup_index_warp_(0),
        kgroup_index_(0) {
  }

  /// advance a threadblock tile
  template <typename WarpLoadIterator>
  CUTLASS_DEVICE void operator++() {
    // is_residual_tile_ = false;
    ++kgroup_index_warp_;
  }

  /// set kgroup index of the warp mma tile
  template <typename WarpLoadIterator>
  CUTLASS_DEVICE void set_kgroup_index(int k_group) {
    using IteratorDetail = Detail<WarpLoadIterator>;
    kgroup_index_ = k_group;
    if constexpr (kRoformer) {
      int warp_contiguous_offset =
          this->kgroup_index_warp_ * IteratorDetail::WarpPitchLinearShape::kContiguous;
      int tile_contigous_offset =
          this->kgroup_index_ * IteratorDetail::InstructionPitchLinearShape::kContiguous;
      int thread_congituous_offset =
          IteratorDetail::calc_thread_contiguous_offset(threadIdx.x % 32);
      int contiguous_offset =
          warp_contiguous_offset + tile_contigous_offset + thread_congituous_offset;
      this->position_denominator_ =
          __powf(10000.0f, __fdividef(contiguous_offset & 0xFFFE, kSizePerHead));
    }
  }

  /// load fragment for the threadblock
  template <typename WarpLoadIterator>
  CUTLASS_DEVICE void load(typename Detail<WarpLoadIterator>::Fragment &frag) {
    using IteratorDetail = Detail<WarpLoadIterator>;
    using Fragment = typename Detail<WarpLoadIterator>::Fragment;
    const int thread_contiguous_offset =
        IteratorDetail::calc_thread_contiguous_offset(threadIdx.x % 32);
    const int warp_tile_contigous_offset =
        IteratorDetail::WarpPitchLinearShape::kContiguous * this->kgroup_index_warp_;
    using LoadType = cutlass::AlignedArray<Element, IteratorDetail::kAccessSize>;
    using AccessType = cutlass::Array<Element, IteratorDetail::kAccessSize>;

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < IteratorDetail::kContiguousCount; ++i) {
      const int k_group_tile_offset =
          warp_tile_contigous_offset + i * IteratorDetail::kContiguousDelta;
      int offset = k_group_tile_offset + thread_contiguous_offset;
      LoadType value;
      cutlass::arch::global_load<LoadType, sizeof(LoadType)>(
          value, (void const *)(this->ptr_ + offset), offset < kSizePerHead);
      frag_ptr[i] = *reinterpret_cast<AccessType *>(&value);
    }
  }

  /// transform mma frag
  template <typename WarpLoadIterator>
  CUTLASS_DEVICE typename WarpLoadIterator::Fragment operator()(
      typename WarpLoadIterator::Fragment const &mma_frag,
      typename Detail<WarpLoadIterator>::Fragment const &frag) const {
    using IteratorDetail = Detail<WarpLoadIterator>;
    using FragmentMma = typename WarpLoadIterator::Fragment;
    FragmentMma ret;
    static_assert(
        IteratorDetail::kStridedCount * IteratorDetail::kAccessSize == FragmentMma::kElements,
        "kElements of FragmentMma not expected");
    using BiasType = cutlass::Array<Element, IteratorDetail::kAccessSize>;
    BiasType bias = reinterpret_cast<BiasType const *>(&frag)[this->kgroup_index_];
    auto ret_ptr = reinterpret_cast<BiasType *>(&ret);
    auto mma_frag_ptr = reinterpret_cast<BiasType const *>(&mma_frag);

    // used to check access in range
    int warp_strided_offset =
        IteratorDetail::kIsOperandA ? this->warp_offsets_.row() : this->warp_offsets_.column();

    int thread_strided_offset = IteratorDetail::calc_thread_strided_offset(threadIdx.x % 32);

    int strided_offset = warp_strided_offset + thread_strided_offset;
    int valid_threshold = (valid_seqlen_ - strided_offset + (IteratorDetail::kStridedDelta - 1)) /
                          IteratorDetail::kStridedDelta;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < IteratorDetail::kStridedCount; ++i) {
      if (i < valid_threshold) {
        cutlass::plus<BiasType> plus_op;
        ret_ptr[i] = plus_op(mma_frag_ptr[i], bias);

        if constexpr (kRoformer) {
          int strided_offset_i = strided_offset + i * IteratorDetail::kStridedDelta;
          float position_enc = __fdividef(strided_offset_i, this->position_denominator_);
          float state0 = static_cast<float>(ret_ptr[i][0]);
          float state1 = static_cast<float>(ret_ptr[i][1]);

          float sin_pos = __sinf(position_enc);
          float cos_pos = __cosf(position_enc);

          ret_ptr[i][0] = static_cast<Element>(state0 * cos_pos - state1 * sin_pos);
          ret_ptr[i][1] = static_cast<Element>(state1 * cos_pos + state0 * sin_pos);
        }
      } else {
        ret_ptr[i].clear();
      }
    }

    return ret;
  }
};

/// Add Bias to Value
template <typename Element_, int SizePerHead>
class ValueBiasAddOp {
 public:
  using Element = Element_;
  static constexpr int kSizePerHead = SizePerHead;

 public:
  template <typename WarpLoadIterator>
  struct Detail {
    using WarpShape = typename WarpLoadIterator::Shape;
    using InstructionShape = typename WarpLoadIterator::InstructionShape;
    static_assert(WarpLoadIterator::kOperand == cutlass::gemm::Operand::kB,
                  "Value bias must be B operand");
    static_assert(InstructionShape::kColumn == 8, "Only support 8 Instruction Column");

    static constexpr int kIterations = WarpShape::kColumn / InstructionShape::kColumn;
    static constexpr int kDelta = 8;
    static constexpr int kAccessSize = 2;

    using Fragment = cutlass::Array<Element, kIterations>;

    static constexpr int calc_thread_row_offset(int lane_id) {
      return (lane_id & 0x3) * kAccessSize;
    }

    static constexpr int calc_thread_column_offset(int lane_id) {
      return lane_id >> 2;
    }
  };

  struct Params {
    Element const *ptr;
    int batch_stride;
    int seqlen = 0;
    int const *seqlen_offsets = nullptr;
  };

 private:
  Element const *ptr_;
  int const warp_column_offset_;
  int const extent_row_;
  int const valid_seqlen_;

  int kgroup_index_warp_;
  int kgroup_index_;

 public:
  CUTLASS_HOST_DEVICE
  ValueBiasAddOp(Params const &params, int problem_idx, int batch_idx, cutlass::MatrixCoord extent,
                 cutlass::MatrixCoord warp_offsets)
      : ptr_(params.ptr + batch_idx * params.batch_stride),
        warp_column_offset_(warp_offsets.column()),
        extent_row_(extent.row()),
        valid_seqlen_(params.seqlen_offsets == nullptr
                          ? params.seqlen
                          : __ldg(&params.seqlen_offsets[problem_idx + 1]) -
                                __ldg(&params.seqlen_offsets[problem_idx])),
        kgroup_index_warp_(0),
        kgroup_index_(0) {
  }

  /// advance a threadblock tile
  template <typename WarpLoadIterator>
  CUTLASS_DEVICE void operator++() {
    ++kgroup_index_warp_;
  }

  /// set kgroup index of the warp mma tile
  template <typename WarpLoadIterator>
  CUTLASS_DEVICE void set_kgroup_index(int k_group) {
    kgroup_index_ = k_group;
  }

  /// load fragment for the threadblock
  template <typename WarpLoadIterator>
  CUTLASS_DEVICE void load(typename Detail<WarpLoadIterator>::Fragment &frag) {
    if (this->kgroup_index_warp_ > 0) {
      return;
    }
    using IteratorDetail = Detail<WarpLoadIterator>;
    using Fragment = typename IteratorDetail::Fragment;

    int thread_column_offset =
        this->warp_column_offset_ + IteratorDetail::calc_thread_column_offset(threadIdx.x % 32);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < IteratorDetail::kIterations; ++i) {
      const int instruction_offset = i * IteratorDetail::kDelta;
      int offset = instruction_offset + thread_column_offset;
      frag[i] = this->ptr_[offset];
    }
  }

  /// transform mma frag
  template <typename WarpLoadIterator>
  CUTLASS_DEVICE typename WarpLoadIterator::Fragment operator()(
      typename WarpLoadIterator::Fragment const &mma_frag,
      typename Detail<WarpLoadIterator>::Fragment const &frag) const {
    using IteratorDetail = Detail<WarpLoadIterator>;
    using FragmentMma = typename WarpLoadIterator::Fragment;
    using Fragment = typename Detail<WarpLoadIterator>::Fragment;
    static_assert(FragmentMma::kElements == Fragment::kElements * IteratorDetail::kAccessSize,
                  "FragmentMma is expected to have exactly kAccessSize times more elements "
                  "thant Fragment");
    FragmentMma ret;

    int residual = extent_row_ % IteratorDetail::WarpShape::kRow;
    residual = residual ? residual : IteratorDetail::WarpShape::kRow;

    int thread_row_offset = IteratorDetail::calc_thread_row_offset(threadIdx.x % 32);
    int instruction_row_offset = kgroup_index_ * IteratorDetail::InstructionShape::kRow;
    int valid_threshold;

    if (this->kgroup_index_warp_ == 0) {
      valid_threshold =
          std::min(residual, valid_seqlen_) - instruction_row_offset - thread_row_offset;
    } else {
      int threadblock_row_offset =
          residual + (this->kgroup_index_warp_ - 1) * IteratorDetail::WarpShape::kRow;
      valid_threshold =
          valid_seqlen_ - threadblock_row_offset - instruction_row_offset - thread_row_offset;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < IteratorDetail::kAccessSize; ++i) {
      if (i < valid_threshold) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < IteratorDetail::kIterations; ++j) {
          ret[j * IteratorDetail::kAccessSize + i] =
              mma_frag[j * IteratorDetail::kAccessSize + i] + frag[j];
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < IteratorDetail::kIterations; ++j) {
          ret[j * IteratorDetail::kAccessSize + i] = Element(0);
        }
      }
    }
    return ret;
  }
};

}  // namespace cutlass_ops
}  // namespace bytetransformer
