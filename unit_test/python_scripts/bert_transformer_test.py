# Copyright 2023 Bytedance Ltd. and/or its affiliates.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import argparse
import timeit
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


random.seed(4)
np.random.seed(3)
torch_gen = torch.manual_seed(2)
torch.cuda.manual_seed(1)


def transpose_for_scores(x, n_heads, head_size):
    # (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
    new_x_shape = x.size()[:-1] + (n_heads, head_size)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)


def seqlen_to_mask(lengths, max_len):
    batch_size = lengths.numel()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    return mask


def set_dtype(ts: torch.Tensor, dtype: str):
    if dtype == "fp32":
        return ts.float()
    elif dtype == "fp16":
        return ts.half()
    raise RuntimeError(f"Unsupported dtype {dtype}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_size", type=int)
    parser.add_argument("seqlen", type=int)
    parser.add_argument("head_num", type=int)
    parser.add_argument("head_size", type=int)
    parser.add_argument("--n_layers", default=1, type=int, help="number of transformer layers")
    parser.add_argument("--avg_seqlen", default=0, type=int, help="average seqlen, 0 for equal to seqlen")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--export_data", action="store_true", help="whether to export test data. "
                                                                   "if true, only run pytorch to generate data, don't run ByteTransformer")
    parser.add_argument("--export_data_path", default="./", type=str, help="path to export test data")
    parser.add_argument("--lib_path", default="./lib/libths_bytetransformer.so",
                        type=str, help="lib path of torch op ext")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    seqlen = args.seqlen
    head_num = args.head_num
    head_size = args.head_size
    avg_seqlen = args.avg_seqlen
    if avg_seqlen <= 0:
        avg_seqlen = seqlen
    n_layers = args.n_layers
    hidden_dim = head_num * head_size
    dtype = args.dtype
    export_data = args.export_data
    export_data_path = args.export_data_path
    lib_path = args.lib_path
    iters = args.iters

    low, high = (2 * avg_seqlen - seqlen, seqlen + 1) if 2 * avg_seqlen > seqlen else (0, 2 * avg_seqlen + 1)
    input_lens = torch.randint(low=low, high=high, size=(batch_size,))
    print("input_lengths:", input_lens)

    seqlen_mask = seqlen_to_mask(input_lens, seqlen)

    # autopep8: off
    qkv_kernel                  = [set_dtype(torch.empty(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(n_layers)]
    qkv_bias                    = [set_dtype(torch.empty(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(n_layers)]
    attr_output_kernel          = [set_dtype(torch.empty(hidden_dim, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(n_layers)]
    attr_output_bias            = [set_dtype(torch.empty(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(n_layers)]
    attr_output_layernorm_gamma = [set_dtype(torch.empty(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(n_layers)]
    attr_output_layernorm_beta  = [set_dtype(torch.empty(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(n_layers)]
    inter_kernel                = [set_dtype(torch.empty(hidden_dim, hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(n_layers)]
    inter_bias                  = [set_dtype(torch.empty(hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(n_layers)]
    output_kernel               = [set_dtype(torch.empty(hidden_dim * 4, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(n_layers)]
    output_bias                 = [set_dtype(torch.empty(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(n_layers)]
    output_layernorm_gamma      = [set_dtype(torch.empty(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(n_layers)]
    output_layernorm_beta       = [set_dtype(torch.empty(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(n_layers)]
    from_tensor                 = set_dtype(torch.empty(batch_size, seqlen, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)

    attr_mask                   = set_dtype(torch.tile(seqlen_mask, dims=(seqlen,)).reshape(batch_size, seqlen, seqlen).cuda(), dtype)
    # autopep8: on
    is_remove_padding = True
    use_fused_attention = True
    transformer_output = [None for _ in range(n_layers)]

    with torch.no_grad():
        hidden_states = from_tensor
        for layer in range(n_layers):
            input_tensor = hidden_states
            qkv = torch.matmul(hidden_states, qkv_kernel[layer]) + qkv_bias[layer]

            q, k, v = qkv.chunk(3, dim=-1)
            q = transpose_for_scores(q, head_num, head_size)
            k = transpose_for_scores(k, head_num, head_size)
            v = transpose_for_scores(v, head_num, head_size)

            # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_size ** .5)

            scores -= 10000.0 * (1.0 - attr_mask.unsqueeze(1))

            probs = F.softmax(scores, dim=-1)
            # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
            h = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
            # -merge-> (B, S, D)
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            hidden_states = h.view(new_context_layer_shape)

            hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]

            hidden_states = hidden_states + input_tensor
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
                                         weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
            residual = hidden_states

            hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer]
            hidden_states = F.gelu(hidden_states)
            hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]

            hidden_states = hidden_states + residual
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
                                         weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])

            transformer_output[layer] = hidden_states

    if export_data:
        masked_output = transformer_output[0]
        masked_output = masked_output * set_dtype(seqlen_mask.unsqueeze(-1).cuda(), dtype)

        all_vars = [qkv_kernel[0], qkv_bias[0],
                    attr_output_kernel[0], attr_output_bias[0], attr_output_layernorm_gamma[0], attr_output_layernorm_beta[0],
                    inter_kernel[0], inter_bias[0],
                    output_kernel[0], output_bias[0], output_layernorm_gamma[0], output_layernorm_beta[0],
                    from_tensor, attr_mask, masked_output
                    ]
        file_list = ["qkv_kernel.in", "qkv_bias.in",
                     "attr_output_kernel.in", "attr_output_bias.in", "attr_output_layernorm_gamma.in", "attr_output_layernorm_beta.in",
                     "inter_kernel.in", "inter_bias.in",
                     "output_kernel.in", "output_bias.in", "output_layernorm_gamma.in", "output_layernorm_beta.in",
                     "from_tensor.in", "atten_mask.in", "transformer_out.out"
                     ]
        idx = 0
        for var in all_vars:
            print(str(idx) + " " + file_list[idx] + " " +
                  str(var.shape) + " " + str(var.dtype))
            np.savetxt(file_list[idx], set_dtype(var.flatten(), "fp32").cpu().numpy(), delimiter='\n', fmt='%f')
            idx = idx + 1
    else:
        torch.ops.load_library(lib_path)
        warmup_iters = 5
        for i in range(warmup_iters + iters):
            if i == warmup_iters:
                t0 = timeit.default_timer()

            hidden_states = from_tensor

            for layer in range(n_layers):
                hidden_states = torch.ops.ByteTransformer.BertTransformer(
                    head_num, head_size,
                    qkv_kernel[layer], qkv_bias[layer],
                    attr_output_kernel[layer], attr_output_bias[layer],
                    attr_output_layernorm_gamma[layer], attr_output_layernorm_beta[layer],
                    inter_kernel[layer], inter_bias[layer], output_kernel[layer], output_bias[layer],
                    output_layernorm_gamma[layer], output_layernorm_beta[layer],
                    hidden_states, attr_mask,
                    is_remove_padding, use_fused_attention)

            output = hidden_states

        t1 = timeit.default_timer()
        masked_output = transformer_output[-1]
        masked_output = masked_output * set_dtype(seqlen_mask.unsqueeze(-1).cuda(), dtype)
        print("max diff:", torch.max(torch.abs(masked_output - output)).cpu())
        print("time costs:    {:.2f} ms".format((t1 - t0) * 1000 / iters))
