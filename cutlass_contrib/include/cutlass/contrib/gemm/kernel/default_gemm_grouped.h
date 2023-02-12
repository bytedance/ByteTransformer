/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This file is copied from NVIDIA/cutlass and modifed.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.

      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.
*/
#pragma once

#include "../kernel/gemm_grouped.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/contrib/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/contrib/args_pack_def.h"
#include "cutlass/contrib/gemm/threadblock/default_mma.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contrib {
namespace gemm {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Tensor defs
    typename TensorDefA, typename TensorDefB, typename TensorDefC, typename TensorDefD,
    /// Prolog defs
    typename PrologueDefA, typename PrologueDefB,
    // Epilogue defs
    typename EpilogueDef,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Tile shape def
    typename TileShapeDef,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Operator to generate problem size
    typename ProblemSizeOperator,
    /// Operator to generate batch count
    typename BatchCountOperator>
struct DefaultGemmGrouped {
  using Mma = typename cutlass::contrib::gemm::threadblock::DefaultMma<
      TensorDefA, TensorDefB, TensorDefC, TensorDefD, PrologueDefA, PrologueDefB, EpilogueDef,
      arch::OpClassTensorOp, ArchTag, TileShapeDef, Stages, Operator>::ThreadblockMma;

  static const int kPartitionsK = TileShapeDef::kPartitionsK;
  using EpilogueOp = typename EpilogueDef::Operator;

  using Epilogue = typename cutlass::contrib::epilogue::threadblock::DefaultEpilogueTensorOp<
      typename TileShapeDef::ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOp,
      EpilogueOp::kCount>::Epilogue;

  using ParamsDef = GemmParamsDef<ProblemSizeOperator, BatchCountOperator,
                                  typename TensorDefA::ParamOp, typename TensorDefB::ParamOp,
                                  typename TensorDefC::ParamOp, typename TensorDefD::ParamOp>;
  /// Define the kernel-level GEMM operator.
  using GemmGroupedKernel = kernel::GemmGrouped<Mma, Epilogue, ParamsDef>;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace contrib
}  // namespace cutlass
