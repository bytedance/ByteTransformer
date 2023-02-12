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
#include "bytetransformer/include/bert_transformer.h"
#include "bytetransformer/include/gemm.h"
#include "bytetransformer/include/gemm_bias_act.h"
#include "bytetransformer/include/layernorm.h"
#include "bytetransformer/include/remove_padding.h"

namespace bytetransformer {
template <OperationType OpType>
void BertTransformer<OpType>::bert_infer(
    BertTransformerInferParam infer_param) {
  const DataType_* from_tensor = infer_param.input_tensor;
  const DataType_* atten_mask = infer_param.atten_mask;
  DataType_* transformer_out = infer_param.transformer_output;
  void* buf = infer_param.buf;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;
  cublasHandle_t cublas_handle = infer_param.cublas_handle;
  cudaStream_t stream = infer_param.stream;

  int input_tensor_size = batch_size * head_num_ * seq_len * size_per_head_;

  DataType_* attention_buf = (DataType_*)((uint8_t*)buf + inner_buf_size_);
  DataType_* inner_buf = (DataType_*)buf;

  DataType_* qkv_buf = inner_buf + 0 * input_tensor_size;
  DataType_* attr_out_buf = inner_buf + 3 * input_tensor_size;
  DataType_* attr_matmul_buf = inner_buf + 4 * input_tensor_size;
  DataType_* inter_matmul_buf = inner_buf + 5 * input_tensor_size;

  int valid_word_num = batch_size * seq_len;

  const int hidden_dim_ = head_num_ * size_per_head_;
  int hidden_dim = (OpType == OperationType::HALF)
                       ? (hidden_dim_ / 2)
                       : hidden_dim_;  // for float & half

  ET_Param et_param;
  if (is_remove_padding_) {
    et_param.word_idx =
        (int*)(inter_matmul_buf + param_.intermediate_size * input_tensor_size);
    et_param.batch_idx = et_param.word_idx + batch_size * seq_len;

    build_sequence_length_padding_offset_kernelLauncher(
        atten_mask, et_param.batch_idx, et_param.word_idx, &valid_word_num,
        batch_size, seq_len, stream);

    et_param.valid_word_num = valid_word_num;

    compressBertInput_kernelLauncher(
        from_tensor, transformer_out, et_param.batch_idx, et_param.word_idx,
        valid_word_num, batch_size, hidden_dim, stream);

    from_tensor =
        transformer_out;  // 1. compress from_tensor      -> transformert_out
    DataType_* tmp =
        transformer_out;  // 2. compute  transformert_out -> inner_buf
    transformer_out = inner_buf;
    inner_buf = tmp;  // 3. restore  inner_buf        -> from_tensor (real
                      // transformer_out)
  }

  int m = valid_word_num;
  int k = head_num_ * size_per_head_;
  int n = k;

  dense_layer_kernel_launcher(from_tensor, param_.attr_kernel_QKV, qkv_buf, m,
                              k, n * 3, cublas_handle, stream,
                              param_.cublas_Algo[0]);

  cudaEvent_t beg, end;
  float elapsed_time = 0.0;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  struct AttentionInferParam<DataType_> attention_infer_param {
    qkv_buf, atten_mask, attr_out_buf, attention_buf, batch_size, seq_len,
        cublas_handle, stream, et_param
  };
  attention_infer_param.attention_bias = infer_param.attention_bias;
  attention_layer_->infer(attention_infer_param);

  dense_layer_kernel_launcher(attr_out_buf, param_.attr_output_kernel,
                              attr_matmul_buf, m, k, n, cublas_handle, stream,
                              param_.cublas_Algo[0]);

  add_bias_input_layernorm_kernel_launcher(
      attr_matmul_buf, from_tensor, param_.attr_output_bias,
      param_.attr_output_layernorm_gamma, param_.attr_output_layernorm_beta, m,
      n, hidden_dim, stream, use_fp32_);

  gemm_bias_gelu(attr_matmul_buf, param_.inter_kernel, inter_matmul_buf,
                 param_.inter_bias, m, k, n * param_.intermediate_size, stream,
                 cublas_handle, param_.cublas_Algo[1], arch_);

  dense_layer_kernel_launcher(inter_matmul_buf, param_.output_kernel,
                              transformer_out, m, k * param_.intermediate_size,
                              n, cublas_handle, stream, param_.cublas_Algo[2]);

  if (is_remove_padding_)
    add_bias_input_layernorm_restore_output_kernel_launcher(
        transformer_out, attr_matmul_buf, param_.output_bias,
        param_.output_layernorm_gamma, param_.output_layernorm_beta,
        batch_size * seq_len, n, hidden_dim, stream, use_fp32_, inner_buf,
        et_param.batch_idx, et_param.word_idx, seq_len);
  else
    add_bias_input_layernorm_kernel_launcher(
        transformer_out, attr_matmul_buf, param_.output_bias,
        param_.output_layernorm_gamma, param_.output_layernorm_beta,
        batch_size * seq_len, n, hidden_dim, stream, use_fp32_);
}


template void BertTransformer<OperationType::FP32>::bert_infer(
    BertTransformerInferParam infer_param);
template void BertTransformer<OperationType::HALF>::bert_infer(
    BertTransformerInferParam infer_param);
}  // namespace bytetransformer
