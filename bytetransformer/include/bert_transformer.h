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
#include "attention.h"
#include "common.h"

#if CUTLASS_ATTENTION
#include "cutlass_attention.h"
#endif

namespace bytetransformer {

template <OperationType OpType>
class BertTransformerParam {
 private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

 public:
  AttentionParam<OpType> attention_param;

  const DataType_ *attr_kernel_QKV;

  const DataType_ *attr_output_kernel;
  const DataType_ *attr_output_bias;
  const void *attr_output_layernorm_gamma;
  const void *attr_output_layernorm_beta;

  const DataType_ *inter_kernel;
  const DataType_ *inter_bias;

  const DataType_ *output_kernel;
  const DataType_ *output_bias;
  const void *output_layernorm_gamma;
  const void *output_layernorm_beta;

  int intermediate_size;

  int cublas_Algo[3];

  BertTransformerParam() {
    attr_kernel_QKV = nullptr;

    attr_output_kernel = nullptr;
    attr_output_bias = nullptr;
    attr_output_layernorm_gamma = nullptr;
    attr_output_layernorm_beta = nullptr;

    inter_kernel = nullptr;
    inter_bias = nullptr;

    output_kernel = nullptr;
    output_bias = nullptr;
    output_layernorm_gamma = nullptr;
    output_layernorm_beta = nullptr;

    intermediate_size = 4;

    if (OpType == OperationType::HALF)
      cublas_Algo[0] = 99, cublas_Algo[1] = 99, cublas_Algo[2] = 99;
    else
      cublas_Algo[0] = -1, cublas_Algo[1] = -1, cublas_Algo[2] = -1;
  }
};

template <typename T>
struct BertTransformerInferParam {
  const T *input_tensor;  // [batch_size, seq_len, hidden_dim]
  const T *atten_mask;    // [batch_size, seq_len, seq_len]  [1, 0]
  T *transformer_output;  // [batch_size, seq_len, hidden_dim]
  void *buf;
  int batch_size;
  int seq_len;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
  const T *attention_bias = NULL;         // [head_num, seq_len, seq_len]
  const uint32_t *position_ids = NULL;    // [batch_size, seq_len]
  const uint32_t *token_type_ids = NULL;  // [batch_size, seq_len]
};

template <OperationType OpType>
class BertTransformer {
 private:
  typedef Traits<OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  BertTransformerParam<OpType> param_;
  using BertTransformerInferParam = struct BertTransformerInferParam<DataType_>;

  Attention<OpType> *attention_layer_ = nullptr;

  const int max_batch_size_, max_seq_len_, head_num_, size_per_head_;

  unsigned long long inner_buf_size_;

  bool use_fused_attention_;
  bool is_remove_padding_;
  const bool use_fp32_;  // gamma & beta datatype
  const ModelType model_type_;
  int arch_ = -1;

 public:
  BertTransformer(const int max_batch_size, const int head_num, const int size_per_head,
                  const int max_seq_len, const bool use_fused_attention = true,
                  const bool is_remove_padding = true, const bool use_fp32 = false,
                  const ModelType model_type = ModelType::Bert)
      : max_batch_size_(max_batch_size),
        max_seq_len_(max_seq_len),
        head_num_(head_num),
        size_per_head_(size_per_head),
        use_fused_attention_(use_fused_attention),
        is_remove_padding_(is_remove_padding),
        use_fp32_(use_fp32),
        model_type_(model_type) {
    if (OpType == OperationType::FP32 || size_per_head_ != 64 || max_seq_len_ > 352)
      use_fused_attention_ = false;

    if (max_batch_size_ <= 2)
      is_remove_padding_ = false;

    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    arch_ = major * 10 + minor;
    if (arch_ < 70)
      use_fused_attention_ = false;

    // different performance of fused attention due to bandwidth of different
    // devices
    if (use_fused_attention_ && (max_seq_len_ % 2 == 0)) {
      if (arch_ == 70 && max_seq_len_ > 256)  // V100
        use_fused_attention_ = false;

      else if (arch_ == 80 && max_seq_len_ > 384)  // A100
        use_fused_attention_ = false;

      else if (arch_ == 86 && max_seq_len_ > 256)
        use_fused_attention_ = false;

      else if (arch_ > 86 && max_seq_len_ > 128)
        use_fused_attention_ = false;
    }

#ifdef CUTLASS_ATTENTION
    bool use_cutlass = false;
    using CutlassAttention = cutlass_ops::CutlassAttention<OpType>;
    if (arch_ == 80 && OpType == OperationType::HALF &&
        CutlassAttention::check_model_type_supported(model_type) &&
        CutlassAttention::check_seqlen_supported(max_seq_len)) {
      use_cutlass = true;
    }

    if (use_cutlass) {
      if constexpr (OpType == OperationType::HALF) {
        attention_layer_ =
            new CutlassAttention(max_batch_size_, head_num_, size_per_head_, max_seq_len_,
                                 use_fused_attention_, is_remove_padding_, model_type_);
      } else {
        throw std::logic_error("Only half supported");
      }
    } else {
#else
    {
#endif  // CUTLASS_ATTENTION
      attention_layer_ =
          new Attention<OpType>(max_batch_size, head_num, size_per_head, max_seq_len,
                                use_fused_attention_, is_remove_padding_, model_type_);
    }
  }

  void initialize(BertTransformerParam<OpType> param) {
    param_ = param;
    attention_layer_->initialize(param_.attention_param);
  }

  unsigned long long cal_bufsize() {
    inner_buf_size_ = 0;

    unsigned long long input_tensor_size =
        max_batch_size_ * head_num_ * max_seq_len_ * size_per_head_;
    inner_buf_size_ +=
        5 * input_tensor_size +
        input_tensor_size * param_.intermediate_size;  // inter_matmul_buf_ = input_tensor_size
                                                       // * 4(default)

    inner_buf_size_ *= sizeof(DataType_);

    if (is_remove_padding_)  // batch_idx & word_idx
      inner_buf_size_ += max_batch_size_ * max_seq_len_ * 2 * sizeof(int);

    inner_buf_size_ = ((inner_buf_size_ + 31) >> 5) << 5;  // For 32B memory alignment

    unsigned long long total_buf_size = attention_layer_->cal_bufsize() + inner_buf_size_;

    return total_buf_size;
  }

  void infer(BertTransformerInferParam infer_param) {
    bert_infer(infer_param);
  }

  void bert_infer(BertTransformerInferParam infer_param);

  ~BertTransformer() {
    delete attention_layer_;
  }
};
}  // namespace bytetransformer
