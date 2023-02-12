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
#include <torch/custom_class.h>
#include "bert_transformer_ext.h"
#include "util.h"

namespace bytetransformer {
namespace torch_ths {
using torch::Tensor;

class BertTransformer : public torch::jit::CustomClassHolder {
 public:
  BertTransformer(int64_t head_num, int64_t head_size, Tensor qkv_kernel, Tensor qkv_bias,
                  Tensor attr_output_kernel, Tensor attr_output_bias,
                  Tensor attr_output_layernorm_gamma, Tensor attr_output_layernorm_beta,
                  Tensor inter_kernel, Tensor inter_bias, Tensor output_kernel, Tensor output_bias,
                  Tensor output_layernorm_gamma, Tensor output_layernorm_beta)
      : _st(qkv_kernel.scalar_type()),
        weights{qkv_kernel,
                qkv_bias,
                attr_output_kernel,
                attr_output_bias,
                attr_output_layernorm_gamma,
                attr_output_layernorm_beta,
                inter_kernel,
                inter_bias,
                output_kernel,
                output_bias,
                output_layernorm_gamma,
                output_layernorm_beta} {
    CHECK_INPUT(qkv_kernel, _st);                    // hidden_dim, hidden_dim * 3
    CHECK_INPUT(qkv_bias, _st);                      // hidden_dim * 3
    CHECK_INPUT(attr_output_kernel, _st);            // hidden_dim, hidden_dim
    CHECK_INPUT(attr_output_bias, _st);              // hidden_dim
    CHECK_INPUT_LOOSE(attr_output_layernorm_gamma);  // hidden_dim
    CHECK_INPUT_LOOSE(attr_output_layernorm_beta);   // hidden_dim
    CHECK_INPUT(inter_kernel, _st);                  // 4 * hidden_dim, hidden_dim
    CHECK_INPUT(inter_bias, _st);                    // 4 * hidden_dim
    CHECK_INPUT(output_kernel, _st);                 // hidden_dim, 4 * hidden_dim
    CHECK_INPUT(output_bias, _st);                   // hidden_dim
    CHECK_INPUT_LOOSE(output_layernorm_gamma);       // hidden_dim
    CHECK_INPUT_LOOSE(output_layernorm_beta);        // hidden_dim
    switch (_st) {
      case at::ScalarType::Float:
        btencoder = new torch_ext::BTEncoder<float>(head_num, head_size, weights);
        break;
      case at::ScalarType::Half:
        btencoder = new torch_ext::BTEncoder<half>(head_num, head_size, weights);
        break;
      default:
        throw std::runtime_error("Wrong Tensor type.");
    }
    head_info = torch::empty({2}, torch::dtype(torch::kInt64));
    head_info[0] = head_num;
    head_info[1] = head_size;
  }

  ~BertTransformer() {
    delete btencoder;
  }

  Tensor forward(Tensor input, Tensor attr_mask) {
    return forwardv2(input, attr_mask, true, true);
  }

  Tensor forwardv2(Tensor input, Tensor attr_mask, bool is_remove_padding,
                   bool use_fused_attention) {
    CHECK_INPUT(input, _st);
    CHECK_INPUT(attr_mask, _st);

    auto input_size = input.sizes();
    int batch_size = input_size[0];
    int seq_len = input_size[1];

    auto output = torch::empty_like(input);
    btencoder->forward(batch_size, seq_len, input, attr_mask, output, is_remove_padding,
                       use_fused_attention);
    return output;
  }

  std::vector<Tensor> get_pickle_info() const {
    std::vector<Tensor> tmp(weights);
    tmp.push_back(head_info);
    return tmp;
  }

 private:
  const at::ScalarType _st;
  torch_ext::IBTEncoder *btencoder;
  Tensor head_info;
  std::vector<Tensor> weights;
};

}  // namespace torch_ths
}  // namespace bytetransformer
