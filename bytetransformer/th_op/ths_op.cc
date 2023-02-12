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
#include <torch/custom_class.h>
#include "ths_op.h"

namespace bytetransformer {
namespace torch_ths {
using torch::Tensor;

static auto bertTransformerEncoderTHS =
    torch::jit::class_<BertTransformer>("ByteTransformer", "BertTransformer")
        .def(torch::jit::init<int64_t, int64_t, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                              Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>())
        .def("forward", &BertTransformer::forward)
        .def("forwardv2", &BertTransformer::forwardv2)
        .def_pickle(
            [](const c10::intrusive_ptr<BertTransformer> &self) -> std::vector<Tensor> {
              return self->get_pickle_info();
            },
            [](std::vector<Tensor> state) -> c10::intrusive_ptr<BertTransformer> {
              int head_num = state[12][0].item().to<int>();
              int head_size = state[12][1].item().to<int>();
              return c10::make_intrusive<BertTransformer>(
                  head_num, head_size, state[0], state[1], state[2], state[3], state[4], state[5],
                  state[6], state[7], state[8], state[9], state[10], state[11]);
            });

}  // namespace torch_ths
}  // namespace bytetransformer
