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
#ifndef CUTLASS_ATTENTION
#else
#include "bytetransformer/include/common.h"
#include "bytetransformer/include/cutlass_attention.h"
#include "bytetransformer/include/cutlass_attention_defs.h"
#include "bytetransformer/include/attention_nofused_utils.h"
#include "cutlass/contrib/args_pack_def.h"
#include "cassert"

namespace bytetransformer {
namespace cutlass_ops {

template <typename Gemm, typename DataType, int kMaxThreadblockNumInRow>
void gemm0_and_softmax_reduce_kernel_launcher(DataType *query, DataType *key, DataType *atten_mask,
                                              DataType *qk_output, float *partial_softmax_buf,
                                              float *softmax_reduced_buf, int *seqlen_offsets,
                                              const int batch_size, const int seq_len,
                                              const int head_num, const int size_per_head,
                                              const float tao, const bool is_remove_padding,
                                              const int sm_count, cudaStream_t stream) {
  using GemmKernel = typename Gemm::GemmKernel;
  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  using ElementC = typename GemmKernel::ElementC;
  using ElementD = typename GemmKernel::ElementD;

  using Shape = typename GemmKernel::ThreadblockShape;
  const auto grid_shape = cutlass::gemm::GemmCoord(
      (seq_len + Shape::kM - 1) / Shape::kM, (seq_len + Shape::kN - 1) / Shape::kN, head_num);
  if (grid_shape.n() >= kMaxThreadblockNumInRow) {
    throw std::runtime_error("grid_shape.n(): " + std::to_string(grid_shape.n()) +
                             " exceeds maximum: " + std::to_string(kMaxThreadblockNumInRow));
  }
  static const int max_active_blocks = Gemm::maximum_active_blocks();
  const int tile_count = grid_shape.m() * grid_shape.n() * grid_shape.k();
  const int cta_count = std::min(tile_count, max_active_blocks * sm_count);

  typename GemmKernel::ParamsDef::ProblemSizeOperator::Params problem_size_op{seqlen_offsets,
                                                                              seq_len};
  typename GemmKernel::ParamsDef::BatchCountOperator::Params batch_count_op{head_num};
  typename AttentionTensorParamGeneratorOp<ElementA>::Params param_A_op{
      reinterpret_cast<ElementA *>(query), size_per_head, batch_size * seq_len * size_per_head,
      seq_len * size_per_head, nullptr};
  typename AttentionTensorParamGeneratorOp<ElementB>::Params param_B_op{
      reinterpret_cast<ElementB *>(key), size_per_head, batch_size * seq_len * size_per_head,
      seq_len * size_per_head, nullptr};
  typename AttentionTensorParamGeneratorOp<ElementC>::Params param_C_op{
      reinterpret_cast<ElementC *>(atten_mask), seq_len, 0, seq_len * seq_len, nullptr};
  typename AttentionTensorParamGeneratorOp<ElementD>::Params param_D_op{
      reinterpret_cast<ElementD *>(qk_output), seq_len, seq_len * seq_len,
      seq_len * seq_len * head_num, nullptr};

  ElementC alpha(1.0f / sqrt(size_per_head * 1.0f) / tao);
  ElementC beta(1.0);
  typename GemmKernel::EpilogueOutputOp::Params epilogue{
      alpha, beta, head_num, seq_len, grid_shape.n(), partial_softmax_buf};
  const int problem_count = batch_size;

  auto args = typename Gemm::Arguments(cutlass::gemm::GemmUniversalMode::kBatched, problem_count,
                                       cta_count, problem_size_op, batch_count_op,
                                       {},        // prologue_A, identity
                                       {},        // prolgoue_B, identity
                                       epilogue,  // partial softmax epilogue
                                       param_A_op, param_B_op, param_C_op, param_D_op);
  // launch kernel
  auto gemm = Gemm();
  auto status = gemm.initialize(args, nullptr, stream);
  CUTLASS_CHECK(status);
  status = gemm(stream);
  CUTLASS_CHECK(status);

  // lightweight kernel to calculate the final reduction for softmax
  softmax_reduction_kernel_launcher<float, Shape::kN>(
      partial_softmax_buf, seqlen_offsets, softmax_reduced_buf, batch_size, head_num, seq_len,
      grid_shape.n(), is_remove_padding, stream);
}

template <typename Gemm, typename DataType>
void gemm1_kernel_launcher(DataType *qk_output, DataType *value, DataType *attention_output,
                           float *softmax_reduced_buf, int const *seqlen_offsets,
                           const int batch_size, const int seq_len, const int head_num,
                           const int size_per_head, const bool is_remove_padding,
                           const int sm_count, cudaStream_t stream) {
  using GemmKernel = typename Gemm::GemmKernel;
  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  using ElementC = typename GemmKernel::ElementC;
  using ElementD = typename GemmKernel::ElementD;

  using Shape = typename GemmKernel::ThreadblockShape;
  const auto grid_shape = cutlass::gemm::GemmCoord(
      (seq_len + Shape::kM - 1) / Shape::kM, (seq_len + Shape::kN - 1) / Shape::kN, head_num);
  const int tile_count = grid_shape.m() * grid_shape.n() * grid_shape.k();
  static const int max_active_blocks = Gemm::maximum_active_blocks();
  const int cta_count = std::min(tile_count, max_active_blocks * sm_count);
  const int hidden_dim = head_num * size_per_head;

  typename GemmKernel::ParamsDef::ProblemSizeOperator::Params problem_size_op{seqlen_offsets,
                                                                              seq_len};
  typename GemmKernel::ParamsDef::BatchCountOperator::Params batch_count_op{head_num};

  typename AttentionTensorParamGeneratorOp<ElementA>::Params param_A_op{
      reinterpret_cast<ElementA *>(qk_output), seq_len, seq_len * seq_len,
      seq_len * seq_len * head_num, nullptr};

  typename AttentionTensorParamGeneratorOp<ElementB>::Params param_B_op{
      reinterpret_cast<ElementB *>(value), size_per_head, batch_size * seq_len * size_per_head,
      seq_len * size_per_head, nullptr};
  typename AttentionTensorParamGeneratorOp<ElementC>::Params param_C_op{nullptr, 0, 0, 0};
  typename AttentionTensorParamGeneratorOp<ElementD>::Params param_D_op{
      reinterpret_cast<ElementD *>(attention_output), hidden_dim, size_per_head,
      is_remove_padding ? hidden_dim : seq_len * hidden_dim, seqlen_offsets};

  const int problem_count = batch_size;
  typename GemmKernel::PrologueDefA::Operator::Params prologue_A{head_num, seq_len,
                                                                 softmax_reduced_buf};
  auto args = typename Gemm::Arguments(cutlass::gemm::GemmUniversalMode::kBatched, problem_count,
                                       cta_count, problem_size_op, batch_count_op,
                                       prologue_A,                      // partial softmax prologue
                                       {},                              // identity
                                       {ElementC(1.0), ElementC(0.0)},  // epilogue
                                       param_A_op, param_B_op, param_C_op, param_D_op);
  // launch kernel
  auto gemm = Gemm();
  auto status = gemm.initialize(args, nullptr, stream);
  CUTLASS_CHECK(status);
  status = gemm(stream);
  CUTLASS_CHECK(status);
}

template <OperationType OpType>
template <typename CutlassAttentionCore>
void CutlassAttention<OpType>::infer_impl(AttentionInferParam *infer_param) {
  if ((this->is_remove_padding_ && infer_param->et_param.valid_word_num == 0) ||
      infer_param->batch_size == 0) {
    // early exit if no need to compute
    return;
  }

  void *buf = infer_param->buf;
  const int batch_size = infer_param->batch_size;
  const int seq_len = infer_param->seq_len;
  const int head_num = this->head_num_;
  const int size_per_head = this->size_per_head_;
  cudaStream_t stream = infer_param->stream;
  ET_Param et_param = infer_param->et_param;

  // calc buf pointers
  auto query = (DataType_ *)((uint8_t *)buf);
  auto key = (DataType_ *)((uint8_t *)query + this->buf_sizes_.input_tensor_size);
  auto value = (DataType_ *)((uint8_t *)key + this->buf_sizes_.input_tensor_size);
  auto qk_output = (DataType_ *)((uint8_t *)value + this->buf_sizes_.input_tensor_size);
  auto partial_softmax_buf = (float *)((uint8_t *)qk_output + this->buf_sizes_.qk_output);
  auto softmax_reduced_buf =
      (float *)((uint8_t *)partial_softmax_buf + this->buf_sizes_.partial_softmax);

  const bool is_roformer = false;
  // add bias
  if constexpr (true) {
    dim3 grid, block;

    const int size_per_head_half =
        (OpType == OperationType::HALF) ? size_per_head / 2 : size_per_head;  // Be careful.
    // [batch_size, seq_len, hidden_dim] -> [head_num, batch_size, seq_len, size_per_head]
    grid.x = seq_len, grid.y = batch_size;
    block.x = head_num * size_per_head_half;
    if (this->is_remove_padding_) {
      add_QKV_bias_padding<<<grid, block, 0, stream>>>(  // restore & clean zero for batch_gemm
          infer_param->qkv, this->param_.attr_bias_QKV, query, key, value, batch_size, seq_len,
          head_num, size_per_head_half, is_roformer, et_param.batch_idx, et_param.word_idx);
    } else {
      add_QKV_bias<<<grid, block, 0, stream>>>(infer_param->qkv, this->param_.attr_bias_QKV, query,
                                               key, value, batch_size, seq_len, head_num,
                                               size_per_head_half, is_roformer);
    }
  }

  DataType_ *atten_mask_noconst = const_cast<DataType_ *>(infer_param->atten_mask);
  int *seqlen_offsets = this->is_remove_padding_ ? et_param.batch_idx : nullptr;

  gemm0_and_softmax_reduce_kernel_launcher<typename CutlassAttentionCore::Gemm0, DataType_,
                                           kMaxThreadblockNumInRow>(
      query, key, atten_mask_noconst, qk_output, partial_softmax_buf, softmax_reduced_buf,
      seqlen_offsets, batch_size, seq_len, head_num, size_per_head, this->param_.tao,
      this->is_remove_padding_, this->multi_processor_count_, stream);

  gemm1_kernel_launcher<typename CutlassAttentionCore::Gemm1, DataType_>(
      qk_output, value, infer_param->attention_output, softmax_reduced_buf, seqlen_offsets,
      batch_size, seq_len, head_num, size_per_head, this->is_remove_padding_,
      this->multi_processor_count_, stream);
}

template <OperationType OpType>
template <int SeqLen, int SizePerHead>
void CutlassAttention<OpType>::do_infer(AttentionInferParam *infer_param) {
#define INFER_WITH_KNOWN_CORE                                                               \
  if (this->max_seq_len_ % CutlassAttentionCore::kAlignment) {                              \
    throw std::runtime_error("[ERROR][BT] alignment requirement not satisfied: " +          \
                             std::to_string(this->max_seq_len_) + " is not divisible by " + \
                             std::to_string(CutlassAttentionCore::kAlignment));             \
  }                                                                                         \
  this->infer_impl<CutlassAttentionCore>(infer_param);

#define ATTENTION_CORE_DEF(ARCH, MODEL_TYPE) \
  using CutlassAttentionCore = CutlassAttentionCore<SeqLen, SizePerHead, ARCH, 2, MODEL_TYPE>;

#define INFER_WITH_KNOWN_ARCH(ARCH)                                                          \
  if (this->model_type_ == ModelType::Bert) {                                                \
    ATTENTION_CORE_DEF(ARCH, ModelType::Bert)                                                \
    INFER_WITH_KNOWN_CORE                                                                    \
  } else {                                                                                   \
    throw std::runtime_error("[ERROR][BT] model_type " + std::to_string(this->model_type_) + \
                             " not supported");                                              \
  }

  if (arch_ == 70) {
    throw std::runtime_error("[ERROR][BT] Cutlass attention Sm70 not enabled.");
  } else if (arch_ == 75) {
    throw std::runtime_error("[ERROR][BT] Cutlass attention Sm75 not enabled.");
  } else if (arch_ >= 80) {
    INFER_WITH_KNOWN_ARCH(cutlass::arch::Sm80)
  } else {
    throw std::runtime_error("[ERROR][BT] Invalid arch: " + std::to_string(arch_));
  }

#undef INFER_WITH_KNOWN_ARCH
#undef ATTENTION_CORE_DEF
#undef INFER_WITH_KNOWN_CORE
}

template <OperationType OpType>
void CutlassAttention<OpType>::infer(AttentionInferParam infer_param) {
  static_assert(OpType == OperationType::HALF, "OpType must be HALF");
  static_assert((__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__) >= 1106,
                "NVCC must be newer than 11.6");

  int cur_seq_len = infer_param.seq_len;

  if (cur_seq_len <= 1024) {
    this->do_infer<1024, 64>(&infer_param);
    return;
  }

  printf("[ERROR][exec] unsupport seq_len!\n");
  exit(-1);
}

template class CutlassAttention<OperationType::HALF>;
}  // namespace cutlass_ops
}  // namespace bytetransformer

#endif  // #ifnde CUTLASS_ATTENTION
