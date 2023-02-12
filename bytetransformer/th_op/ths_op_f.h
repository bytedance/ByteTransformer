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
#include "util.h"

namespace bytetransformer {
namespace torch_ths {
using torch::Tensor;

Tensor TransformerEncoder(int64_t head_num, int64_t head_size, Tensor qkv_kernel, Tensor qkv_bias,
                          Tensor attr_output_kernel, Tensor attr_output_bias,
                          Tensor attr_output_layernorm_gamma, Tensor attr_output_layernorm_beta,
                          Tensor inter_kernel, Tensor inter_bias, Tensor output_kernel,
                          Tensor output_bias, Tensor output_layernorm_gamma,
                          Tensor output_layernorm_beta, Tensor input, Tensor attr_mask,
                          bool is_remove_padding, bool use_fused_attention);

}  // namespace torch_ths
}  // namespace bytetransformer
