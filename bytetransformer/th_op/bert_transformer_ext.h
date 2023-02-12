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
#include "bytetransformer/include/common.h"
#include "bytetransformer/include/bert_transformer.h"
#include "traits.h"
#include "util.h"

namespace bytetransformer {
namespace torch_ext {
using torch::Tensor;

class IBTEncoder {
 public:
  virtual ~IBTEncoder() {
  }
  virtual void forward(int batch_size, int seq_len, Tensor &input, Tensor &attr_mask,
                       Tensor &output, bool is_remove_padding, bool use_fused_attention,
                       const ModelType model_type = ModelType::Bert,
                       const Tensor attention_bias = Tensor()) = 0;
};

template <typename T>
class BTEncoder : public IBTEncoder {
 public:
  BTEncoder(int head_num, int head_size, const std::vector<Tensor> &w, int algo_attr_linear = -1,
            int algo_ffn_inter = -1, int algo_ffn_output = -1, int algo_attr_bgemm1 = -1,
            int algo_attr_bgemm2 = -1)
      : _head_num(head_num), _head_size(head_size), _use_fp32(false), _weights(w) {
    encoder_param.attr_kernel_QKV = get_ptr<T>(_weights[0]);
    encoder_param.attention_param.attr_bias_QKV = get_ptr<T>(_weights[1]);
    encoder_param.attr_output_kernel = get_ptr<T>(_weights[2]);
    encoder_param.attr_output_bias = get_ptr<T>(_weights[3]);
    encoder_param.attr_output_layernorm_gamma = get_ptr<void>(_weights[4]);
    encoder_param.attr_output_layernorm_beta = get_ptr<void>(_weights[5]);
    encoder_param.inter_kernel = get_ptr<T>(_weights[6]);
    encoder_param.inter_bias = get_ptr<T>(_weights[7]);
    encoder_param.output_kernel = get_ptr<T>(_weights[8]);
    encoder_param.output_bias = get_ptr<T>(_weights[9]);
    encoder_param.output_layernorm_gamma = get_ptr<void>(_weights[10]);
    encoder_param.output_layernorm_beta = get_ptr<void>(_weights[11]);

    if (THTraits<T>::OpType == OperationType::HALF) {
      if (_weights[10].scalar_type() == at::ScalarType::Float) {
        _use_fp32 = true;
      }
      encoder_param.cublas_Algo[0] = -1 == algo_attr_linear ? 99 : algo_attr_linear;
      encoder_param.cublas_Algo[1] = -1 == algo_ffn_inter ? 99 : algo_ffn_inter;
      encoder_param.cublas_Algo[2] = -1 == algo_ffn_output ? 99 : algo_ffn_output;
      encoder_param.attention_param.cublas_Algo[0] =
          -1 == algo_attr_bgemm1 ? 99 : algo_attr_bgemm1;
      encoder_param.attention_param.cublas_Algo[1] =
          -1 == algo_attr_bgemm2 ? 99 : algo_attr_bgemm2;
    } else {
      encoder_param.cublas_Algo[0] = algo_attr_linear;
      encoder_param.cublas_Algo[1] = algo_ffn_inter;
      encoder_param.cublas_Algo[2] = algo_ffn_output;
      encoder_param.attention_param.cublas_Algo[0] = algo_attr_bgemm1;
      encoder_param.attention_param.cublas_Algo[1] = algo_attr_bgemm2;
    }
  }

  void forward(int batch_size, int seq_len, Tensor &input, Tensor &attr_mask, Tensor &output,
               bool is_remove_padding = true, bool use_fused_attention = true,
               const ModelType model_type = ModelType::Bert,
               const Tensor attention_bias = Tensor()) override {
    CHECK_SEQ_LEN(seq_len)

    BertTransformer<THTraits<T>::OpType> *encoder = new BertTransformer<THTraits<T>::OpType>(
        batch_size, _head_num, _head_size, seq_len, use_fused_attention, is_remove_padding,
        _use_fp32, model_type);
    encoder->initialize(encoder_param);

    const T *from_tensor = get_ptr<T>(input);
    const T *atten_mask = get_ptr<T>(attr_mask);
    T *transformer_out = get_ptr<T>(output);

    at::cuda::CUDAGuard device_guard{input.device().index()};

    unsigned long long buf_size = encoder->cal_bufsize();
    auto buf_tensor =
        torch::empty({(long int)buf_size}, torch::dtype(torch::kInt8).device(torch::kCUDA));
    void *buf = get_ptr<void>(buf_tensor);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    struct BertTransformerInferParam<T> infer_param {
      from_tensor, atten_mask, transformer_out, buf, batch_size, seq_len, cublas_handle, stream
    };
    infer_param.attention_bias = attention_bias.defined() ? get_ptr<T>(attention_bias) : NULL;

    encoder->infer(infer_param);
    delete encoder;
  }

 private:
  const int _head_num, _head_size;
  bool _use_fp32;
  std::vector<Tensor> _weights;
  BertTransformerParam<THTraits<T>::OpType> encoder_param;
};
}  // namespace torch_ext
}  // namespace bytetransformer
