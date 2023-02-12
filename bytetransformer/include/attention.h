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
template <OperationType OpType>
class AttentionParam {
 private:
  using Traits_ = Traits<OpType>;
  using DataType_ = typename Traits_::DataType;

 public:
  const DataType_ *attr_bias_QKV;  // [hidden_dim * 3]
  float tao;

  int cublas_Algo[2];

  AttentionParam() {
    attr_bias_QKV = nullptr;
    tao = 1.0f;

    cublas_Algo[0] = cublas_Algo[1] = Traits_::algo;
  }
};

template <typename T>
struct AttentionInferParam {
  const T *qkv;         // [batch_size, seq_len, hidden_dim * 3]
  const T *atten_mask;  // [batch_size, seq_len, seq_len], [1, 0]
  T *attention_output;  // [batch_size, seq_len, hidden_dim]
  void *buf;
  int batch_size = 0;
  int seq_len = 0;
  cublasHandle_t cublas_handle = nullptr;
  cudaStream_t stream = nullptr;
  ET_Param et_param = {nullptr, nullptr, 0};
  const T *attention_bias = nullptr;         // [head_num, seq_len, seq_len]
  const uint32_t *position_ids = nullptr;    // [batch_size, seq_len]
  const uint32_t *token_type_ids = nullptr;  // [batch_size, seq_len]
};

template <OperationType OpType>
class Attention {
 protected:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;
  const int max_batch_size_, max_seq_len_, head_num_, size_per_head_;
  AttentionParam<OpType> param_;

  using AttentionInferParam = struct AttentionInferParam<DataType_>;

  bool use_fused_attention_;
  const bool is_remove_padding_;
  const ModelType model_type_;

  bool transformer_variety_fuse_flag_;

 public:
  Attention(const int max_batch_size, const int head_num, const int size_per_head,
            const int max_seq_len, const bool use_fused_attention = true,
            const bool is_remove_padding = false, const ModelType model_type = ModelType::Bert)
      : max_batch_size_(max_batch_size),
        max_seq_len_(max_seq_len),
        head_num_(head_num),
        size_per_head_(size_per_head),
        use_fused_attention_(use_fused_attention),
        is_remove_padding_(is_remove_padding),
        model_type_(model_type) {
    if (OpType == OperationType::FP32)
      use_fused_attention_ = false;

    transformer_variety_fuse_flag_ = use_fused_attention_;

    if (model_type_ != ModelType::Bert)
      use_fused_attention_ = false;

    if (transformer_variety_fuse_flag_) {
      if (size_per_head != 64)
        transformer_variety_fuse_flag_ = false;

      if (max_seq_len_ > 256)
        transformer_variety_fuse_flag_ = false;
    }
  }

  void initialize(AttentionParam<OpType> param) {
    param_ = param;
  }

  virtual unsigned long long cal_bufsize() const {
    if (use_fused_attention_)
      return 0;
    else {
      unsigned long long input_tensor_size =
          max_batch_size_ * head_num_ * max_seq_len_ * size_per_head_;
      unsigned long long qk_buf_size =
          ((max_batch_size_ * head_num_ * max_seq_len_ * max_seq_len_ + 15) >> 4)
          << 4;  // for memory alignment
      unsigned long long inner_buf_size = input_tensor_size * 4 + qk_buf_size;
      unsigned long long total_buf_size = inner_buf_size * sizeof(DataType_);
      return total_buf_size;
    }
  }

  virtual void infer(AttentionInferParam infer_param) {
    if (use_fused_attention_) {
      if (infer_param.seq_len <= 80) {
        if (is_remove_padding_)
          fused_rm_infer(infer_param);
        else
          fused_infer(infer_param);
      } else {
        if (is_remove_padding_)
          fused_long_rm_infer(infer_param);
        else
          fused_long_infer(infer_param);
      }
    } else
      nofused_infer(infer_param);
  }

  void nofused_infer(AttentionInferParam infer_param);
  void fused_infer(AttentionInferParam infer_param);
  void fused_rm_infer(AttentionInferParam infer_param);
  void fused_long_infer(AttentionInferParam infer_param);
  void fused_long_rm_infer(AttentionInferParam infer_param);

  virtual ~Attention() {
  }
};
}  // namespace bytetransformer
