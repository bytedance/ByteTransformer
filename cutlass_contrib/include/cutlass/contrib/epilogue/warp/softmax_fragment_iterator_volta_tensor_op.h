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
    \brief This defines a "fragment" iterator for visiting the fragments of an accumulator tile
      that participate in one warp-level store operation.

      Typically, the accumulator tile is the largest single block of register-backed storage
      within the kernel. Storing it to memory is best accomplished by partitioning it into
      smaller tiles and storing these sequentially.

      Round trips through shared memory during the Epilogue phase require partitioning, as
      shared memory capacity is typically insufficient for a threadblock's total accumulator
      size.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/epilogue/warp/volta_tensor_op_policy.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/device_dump.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contrib {
namespace epilogue {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Volta 16x16x4
template <typename WarpShape_,         ///< shape of the warp-level GEMM tile
          typename OperatorElementC_,  ///< matrix multiply operation data type (concept: data
                                       ///< type)
          typename OperatorFragmentC_  ///< matrix multiply operation fragment (concept: Array)
          >
class SoftmaxFragmentIteratorTensorOp<WarpShape_, cutlass::gemm::GemmShape<16, 16, 4>,
                                      OperatorElementC_, OperatorFragmentC_, layout::RowMajor> {
 public:
  using WarpShape = WarpShape_;
  using OperatorShape = cutlass::gemm::GemmShape<16, 16, 4>;
  using InterleavedTileShape = cutlass::gemm::GemmShape<32, 32, 4>;
  using OperatorElementC = OperatorElementC_;
  using OperatorFragmentC = OperatorFragmentC_;
  using Layout = layout::RowMajor;

  using Policy = cutlass::epilogue::warp::VoltaTensorOpPolicy<WarpShape, InterleavedTileShape,
                                                              OperatorElementC_, Layout>;
  using TileIterations = typename Policy::TileIterations;

  // using Fragment = Array<OperatorElementC,
  //                        Policy::TileIterations::kColumn * Policy::MmaIterations::kColumn *
  //                            Policy::kElementsPerMma>;

  using Fragment = typename Policy::Fragment;
  using AccumulatorTile = typename Policy::AccumulatorTile;

  using OutputAccumulatorTile = AccumulatorTile;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

 private:
  /// Internal access type
  using AccessType = typename Policy::AccessType;

 private:
  //
  // Data members
  //

  /// Accumulator tile
  AccessType *accumulators_;

  /// Internal index
  int index_;

 public:
  /// Constructs an iterator
  CUTLASS_HOST_DEVICE
  SoftmaxFragmentIteratorTensorOp(AccumulatorTile &accum)
      : accumulators_(reinterpret_cast<AccessType *>(&accum)), index_(0) {
  }

  /// Increments
  CUTLASS_HOST_DEVICE
  SoftmaxFragmentIteratorTensorOp &operator++() {
    ++index_;
    return *this;
  }

  /// Decrements
  CUTLASS_HOST_DEVICE
  SoftmaxFragmentIteratorTensorOp &operator--() {
    --index_;
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, int index_offset = 0) const {
    int index = index_ + index_offset;

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    // if (threadIdx.x == 0) {
    //   // printf("%d\n", cutlass::MatrixShape<1, 4>::kColumn);
    //   printf("%d %d %d, %d\n",
    //          Fragment::kElements,
    //          TileIterations::kRow,
    //          TileIterations::kColumn,
    //          Policy::kAccessesPerInterleavedTile);
    // }

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::TileIterations::kColumn; ++n) {
      int base_offset =
          (n * Policy::TileIterations::kRow + index / 2) * Policy::kAccessesPerInterleavedTile * 2;
      // int base_offset = (index / 2 * Policy::TileIterations::kColumn + n) *
      //                   Policy::kAccessesPerInterleavedTile * 2;
      int offset0, offset1;
      if (index & 0x1) {
        // mma 1, 3
        offset0 = base_offset + 2;
        offset1 = base_offset + 6;
      } else {
        // mma 0, 2
        offset0 = base_offset;
        offset1 = base_offset + 4;
      }
      frag_ptr[n * 4] = accumulators_[offset0];
      frag_ptr[n * 4 + 1] = accumulators_[offset1];
      frag_ptr[n * 4 + 2] = accumulators_[offset0 + 1];
      frag_ptr[n * 4 + 3] = accumulators_[offset1 + 1];
    }
  }
  /// Stores a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void store(Fragment &frag, int index_offset = 0) const {
    int index = index_ + index_offset;

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::TileIterations::kColumn; ++n) {
      int base_offset =
          (n * Policy::TileIterations::kRow + index / 2) * Policy::kAccessesPerInterleavedTile * 2;
      // int base_offset = (index / 2 * Policy::TileIterations::kColumn + n) *
      //                   Policy::kAccessesPerInterleavedTile * 2;
      int offset0, offset1;
      if (index & 0x1) {
        // mma 1, 3
        offset0 = base_offset + 2;
        offset1 = base_offset + 6;
      } else {
        // mma 0, 2
        offset0 = base_offset;
        offset1 = base_offset + 4;
      }
      accumulators_[offset0] = frag_ptr[n * 4];
      accumulators_[offset1] = frag_ptr[n * 4 + 1];
      accumulators_[offset0 + 1] = frag_ptr[n * 4 + 2];
      accumulators_[offset1 + 1] = frag_ptr[n * 4 + 3];
    }
  }
};

}  // namespace warp
}  // namespace epilogue
}  // namespace contrib
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
