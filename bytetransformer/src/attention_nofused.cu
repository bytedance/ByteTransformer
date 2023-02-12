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
#include "bytetransformer/include/attention.h"
#include "bytetransformer/include/attention_nofused_utils.h"
#include "bytetransformer/include/gemm.h"
#include "bytetransformer/include/softmax.h"
#include "bytetransformer/include/variety_attention_fused.h"

namespace bytetransformer {
template <OperationType OpType>
void Attention<OpType>::nofused_infer(AttentionInferParam infer_param) {
  void* buf = infer_param.buf;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;
  cudaStream_t stream = infer_param.stream;
  ET_Param et_param = infer_param.et_param;

  int input_tensor_size = batch_size * head_num_ * seq_len * size_per_head_;
  int qk_buf_size = ((batch_size * head_num_ * seq_len * seq_len + 15) >> 4)
                    << 4;

  DataType_* query = (DataType_*)buf + 0 * input_tensor_size;
  DataType_* key = (DataType_*)buf + 1 * input_tensor_size;
  DataType_* value = (DataType_*)buf + 2 * input_tensor_size;
  DataType_* qk_buf = (DataType_*)buf + 3 * input_tensor_size;
  DataType_* transpose_dst = qk_buf + qk_buf_size;

  int size_per_head_half = (OpType == OperationType::HALF)
                               ? size_per_head_ / 2
                               : size_per_head_;  // Be careful.

  // [batch_size, seq_len, hidden_dim] -> [head_num, batch_size, seq_len,
  // size_per_head]
  dim3 grid, block;
  grid.x = seq_len, grid.y = batch_size;
  block.x = head_num_ *
            (size_per_head_ / 2);  // Process two adjacent values for float/half
  const bool is_roformer = false;
  if (is_remove_padding_)
    add_QKV_bias_padding<<<grid, block, 0, stream>>>(  // restore & clean zero
                                                       // for batch_gemm
        infer_param.qkv, param_.attr_bias_QKV, query, key, value, batch_size,
        seq_len, head_num_, size_per_head_ / 2, is_roformer, et_param.batch_idx,
        et_param.word_idx);
  else
    add_QKV_bias<<<grid, block, 0, stream>>>(
        infer_param.qkv, param_.attr_bias_QKV, query, key, value, batch_size,
        seq_len, head_num_, size_per_head_ / 2, is_roformer);
  grid.y = 1;

  DataType_ alpha =
                (DataType_)(1.0f / sqrtf(size_per_head_ * 1.0f) / param_.tao),
            beta = (DataType_)0.0f;
  bool add_qk_buf = false;

  if (transformer_variety_fuse_flag_)
    variety_attention_fused_infer(
        (const __half*)query, (const __half*)key, (const __half*)value,
        (const __half*)infer_param.atten_mask,
        add_qk_buf ? (const __half*)qk_buf : NULL,
        (const __half*)infer_param.attention_bias,
        (__half*)infer_param.attention_output, head_num_, batch_size, seq_len,
        size_per_head_, (float)alpha, infer_param.stream,
        is_remove_padding_ ? et_param.batch_idx : NULL);
  else {
    cublas_Gemm_Strided_Batched(
        query, key, qk_buf, seq_len, size_per_head_, seq_len,
        head_num_ * batch_size, CUBLAS_OP_N, CUBLAS_OP_T, alpha, beta,
        infer_param.cublas_handle, stream, param_.cublas_Algo[0]);

    if (is_remove_padding_)
      softmax_et_kernelLauncher<OpType, DataType_>(
          qk_buf, infer_param.attention_bias, infer_param.atten_mask,
          batch_size, seq_len, head_num_, stream, et_param.batch_idx,
          et_param.word_idx, et_param.valid_word_num);
    else
      softmax_kernelLauncher<OpType, DataType_>(
          qk_buf, infer_param.attention_bias, infer_param.atten_mask,
          batch_size, seq_len, head_num_, stream);

    alpha = (DataType_)1.0f, beta = (DataType_)0.0f;
    cublas_Gemm_Strided_Batched(
        qk_buf, value, transpose_dst, seq_len, seq_len, size_per_head_,
        head_num_ * batch_size, CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta,
        infer_param.cublas_handle, stream, param_.cublas_Algo[1]);

    block.x = size_per_head_half, block.y = head_num_;
    if (is_remove_padding_)
      transpose_rm_padding<<<et_param.valid_word_num, block, 0, stream>>>(
          transpose_dst, infer_param.attention_output, batch_size, seq_len,
          head_num_, size_per_head_half, et_param.batch_idx, et_param.word_idx);
    else
      transpose<<<batch_size * seq_len, block, 0, stream>>>(
          transpose_dst, infer_param.attention_output, batch_size, seq_len,
          head_num_, size_per_head_half);
  }
}

template void Attention<OperationType::FP32>::nofused_infer(
    AttentionInferParam infer_param);
template void Attention<OperationType::HALF>::nofused_infer(
    AttentionInferParam infer_param);
}  // namespace bytetransformer
