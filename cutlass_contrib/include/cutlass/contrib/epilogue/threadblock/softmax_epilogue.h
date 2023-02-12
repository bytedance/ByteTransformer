/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/functional.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contrib {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator without splitk
template <typename Shape_,            ///< Shape of threadblock tile (concept: GemmShape)
          typename WarpMmaOperator_,  ///< Warp-level MMA operator (concept:
                                      ///< gemm::warp::MmaTensorOp)
          int PartitionsK,            ///< Number of partitions of the K dimension, CTA_K / Warp_K
          typename OutputTileIterator_,  ///< Tile iterator reading and writing output tensors
          typename AccumulatorFragmentIterator_,  ///< Fragment iterator selecting accumulators
          typename OutputOp_                      ///< Output operator
          >
class SoftmaxEpilogue {
 public:
  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using OutputOp = OutputOp_;

  /// Output layout is always row-major
  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;
  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;
  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

 public:
  struct SharedStorage {
    /// empty shared memory
  };

 public:
  /// Constructor
  CUTLASS_DEVICE
  SoftmaxEpilogue() {
  }

  CUTLASS_DEVICE
  void operator()(OutputOp const &output_op,      ///< Output operator
                  AccumulatorTile &accumulators,  ///< Complete warp-level accumulator tile
                  AccumulatorTile &softmax_accumlators,
                  OutputTileIterator source_iterator) {  ///< Threadblock tile coordinate in GEMM
                                                         ///< (in units of threadblock tiles)
    compute_source_needed_(output_op, accumulators, softmax_accumlators, source_iterator);
    // compute_source_no_needed_(output_op, accumulators, softmax_accumlators);
  }

  CUTLASS_DEVICE
  void compute_source_needed_(
      OutputOp const &output_op,      ///< Output operator
      AccumulatorTile &accumulators,  ///< Complete warp-level accumulator tile
      AccumulatorTile &softmax_accumlators,
      OutputTileIterator source_iterator) {  ///< Threadblock tile coordinate in GEMM (in units of
                                             ///< threadblock tiles)

    typename OutputTileIterator::Fragment source_fragment;

    source_fragment.clear();

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    AccumulatorFragmentIterator softmax_fragment_iterator(softmax_accumlators);

    // if (threadIdx.x  == 0){
    // printf("operationcount : column, %d, row : %d, elementper access: %d, Instruct iterations:
    // %d, iterations : %d\n", AccumulatorFragmentIterator::Policy::OperatorCount::kColumn,
    //         AccumulatorFragmentIterator::Policy::OperatorCount::kRow,
    //         AccumulatorFragmentIterator::Policy::kElementsPerAccess,
    //         AccumulatorFragmentIterator::Policy::kIterationsPerInstruction,
    //         AccumulatorFragmentIterator::kIterations);

    // printf("outputtile: iterations: %d \n",
    //       OutputTileIterator::kIterations);
    // }

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      source_iterator.load(source_fragment);
      ++source_iterator;

      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;

      typename AccumulatorFragmentIterator::Fragment softmax_fragment;
      softmax_fragment = output_op(accum_fragment, source_fragment);

      softmax_fragment_iterator.store(softmax_fragment);
      ++softmax_fragment_iterator;
    }
  }

  CUTLASS_DEVICE
  void compute_source_no_needed_(
      OutputOp const &output_op,               ///< Output operator
      AccumulatorTile &accumulators,           ///< Complete warp-level accumulator tile
      AccumulatorTile &softmax_accumlators) {  ///< Threadblock tile coordinate in GEMM (in units
                                               ///< of threadblock tiles)

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    AccumulatorFragmentIterator softmax_fragment_iterator(softmax_accumlators);

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < AccumulatorFragmentIterator::kIterations; ++iter) {
      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;

      typename AccumulatorFragmentIterator::Fragment softmax_fragment;
      softmax_fragment = output_op(accum_fragment);

      softmax_fragment_iterator.store(softmax_fragment);
      ++softmax_fragment_iterator;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace contrib
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
