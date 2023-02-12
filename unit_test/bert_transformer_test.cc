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
#include "helper.h"
using namespace std;
using namespace bytetransformer;

#ifdef FP16
typedef __half T;
static const OperationType OpType = OperationType::HALF;
#else
typedef float T;
static const OperationType OpType = OperationType::FP32;
#endif

int main(int argc, char *argv[]) {
  if (argc < 5) {
    printf(
        "./bert_transformer_test <batch_size> <seq_len> <head_num> <size_per_head>\n");
    return 0;
  }

  int batch_size = atoi(argv[1]);
  int seq_len = atoi(argv[2]);
  int head_num = atoi(argv[3]);
  int size_per_head = atoi(argv[4]);

  int hidden_dim = head_num * size_per_head;

  bool is_remove_padding = true;
  if (batch_size <= 2) {
    is_remove_padding = false;
  }
  bool use_fused_attention = true;

  if (OpType == OperationType::FP32 || size_per_head != 64 || seq_len > 352)
    use_fused_attention = false;

  bool use_fp32 = true;
  if (OpType == OperationType::FP32)
    use_fp32 = true;

  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device %s\n", prop.name);

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));
  cublasHandle_t cublas_handle;
  check_cuda_error(cublasCreate(&cublas_handle));
  check_cuda_error(cublasSetStream(cublas_handle, stream));

  BertTransformerParam<OpType> param;

  FILE *fd = fopen("gemm_config.in", "r");
  int err = 0;
  if (fd == NULL)
    printf("gemm_config.in is not found, using default GEMM algorithms\n");
  else {
    err =
        fscanf(fd, "%d%d%d", &param.cublas_Algo[0], &param.cublas_Algo[1], &param.cublas_Algo[2]);
    err += fscanf(fd, "%d%d", &param.attention_param.cublas_Algo[0],
                  &param.attention_param.cublas_Algo[1]);
    fclose(fd);
    if (err != 5)
      printf("loading GEMM algorithms error, using default GEMM algorithms\n");
  }

  T *qkv_kernel;
  device_malloc(&qkv_kernel, hidden_dim * hidden_dim * 3);
  load_weights(&qkv_kernel, (char *)"qkv_kernel.in", hidden_dim * hidden_dim * 3);
  param.attr_kernel_QKV = qkv_kernel;

  T *qkv_bias;
  device_malloc(&qkv_bias, hidden_dim * 3);
  load_weights(&qkv_bias, (char *)"qkv_bias.in", hidden_dim * 3);
  param.attention_param.attr_bias_QKV = qkv_bias;

  T *attr_output_kernel, *attr_output_bias;
  device_malloc(&attr_output_kernel, hidden_dim * hidden_dim);
  load_weights(&attr_output_kernel, (char *)"attr_output_kernel.in", hidden_dim * hidden_dim);
  device_malloc(&attr_output_bias, hidden_dim);
  load_weights(&attr_output_bias, (char *)"attr_output_bias.in", hidden_dim);
  param.attr_output_kernel = attr_output_kernel;
  param.attr_output_bias = attr_output_bias;

  T *inter_kernel, *inter_bias;
  device_malloc(&inter_kernel, hidden_dim * hidden_dim * 4);
  load_weights(&inter_kernel, (char *)"inter_kernel.in", hidden_dim * hidden_dim * 4);
  device_malloc(&inter_bias, hidden_dim * 4);
  load_weights(&inter_bias, (char *)"inter_bias.in", hidden_dim * 4);
  param.inter_kernel = inter_kernel;
  param.inter_bias = inter_bias;

  T *output_kernel, *output_bias;
  device_malloc(&output_kernel, hidden_dim * 4 * hidden_dim);
  load_weights(&output_kernel, (char *)"output_kernel.in", hidden_dim * 4 * hidden_dim);
  device_malloc(&output_bias, hidden_dim);
  load_weights(&output_bias, (char *)"output_bias.in", hidden_dim);
  param.output_kernel = output_kernel;
  param.output_bias = output_bias;

  if (use_fp32) {
    float *attr_output_layernorm_gamma, *attr_output_layernorm_beta, *output_layernorm_gamma,
        *output_layernorm_beta;
    device_malloc(&attr_output_layernorm_gamma, hidden_dim);
    load_weights(&attr_output_layernorm_gamma, (char *)"attr_output_layernorm_gamma.in",
                 hidden_dim);
    device_malloc(&attr_output_layernorm_beta, hidden_dim);
    load_weights(&attr_output_layernorm_beta, (char *)"attr_output_layernorm_beta.in", hidden_dim);
    device_malloc(&output_layernorm_gamma, hidden_dim);
    load_weights(&output_layernorm_gamma, (char *)"output_layernorm_gamma.in", hidden_dim);
    device_malloc(&output_layernorm_beta, hidden_dim);
    load_weights(&output_layernorm_beta, (char *)"output_layernorm_beta.in", hidden_dim);
    param.attr_output_layernorm_gamma = attr_output_layernorm_gamma;
    param.attr_output_layernorm_beta = attr_output_layernorm_beta;
    param.output_layernorm_gamma = output_layernorm_gamma;
    param.output_layernorm_beta = output_layernorm_beta;
  } else {
    __half *attr_output_layernorm_gamma, *attr_output_layernorm_beta, *output_layernorm_gamma,
        *output_layernorm_beta;
    device_malloc(&attr_output_layernorm_gamma, hidden_dim);
    load_weights(&attr_output_layernorm_gamma, (char *)"attr_output_layernorm_gamma.in",
                 hidden_dim);
    device_malloc(&attr_output_layernorm_beta, hidden_dim);
    load_weights(&attr_output_layernorm_beta, (char *)"attr_output_layernorm_beta.in", hidden_dim);
    device_malloc(&output_layernorm_gamma, hidden_dim);
    load_weights(&output_layernorm_gamma, (char *)"output_layernorm_gamma.in", hidden_dim);
    device_malloc(&output_layernorm_beta, hidden_dim);
    load_weights(&output_layernorm_beta, (char *)"output_layernorm_beta.in", hidden_dim);
    param.attr_output_layernorm_gamma = attr_output_layernorm_gamma;
    param.attr_output_layernorm_beta = attr_output_layernorm_beta;
    param.output_layernorm_gamma = output_layernorm_gamma;
    param.output_layernorm_beta = output_layernorm_beta;
  }

  BertTransformer<OpType> *transformer_layer =
      new BertTransformer<OpType>(batch_size, head_num, size_per_head, seq_len,
                                  use_fused_attention, is_remove_padding, use_fp32);
  transformer_layer->initialize(param);

  void *buf;
  unsigned long long buf_size = transformer_layer->cal_bufsize();
  printf("buf_size %lld bytes, %f in MB\n", buf_size, buf_size / 1024.0 / 1024.0);
  device_malloc((void **)&buf, buf_size);

  T *from_tensor, *atten_mask;
  device_malloc(&from_tensor, batch_size * seq_len * hidden_dim);
  load_weights(&from_tensor, (char *)"from_tensor.in", batch_size * seq_len * hidden_dim);
  device_malloc(&atten_mask, batch_size * seq_len * seq_len);
  load_weights(&atten_mask, (char *)"atten_mask.in", batch_size * seq_len * seq_len);

  T *transformer_out;
  device_malloc(&transformer_out, batch_size * seq_len * hidden_dim);

  struct BertTransformerInferParam<T> infer_param {
    from_tensor, atten_mask, transformer_out, buf, batch_size, seq_len, cublas_handle, stream
  };
  transformer_layer->infer(infer_param);

#ifdef FP16
  printf("\033[0;31m");
  printf("Precision in FP16\n");
  printf("\033[0m");
#else
  printf("\033[0;31m");
  printf("Precision in FP32\n");
  printf("\033[0m");
#endif

  result_check(transformer_out, (char *)"transformer_out.out", batch_size * seq_len * hidden_dim);

  int ite = 50;
  // warm up
  for (int i = 0; i < ite; ++i)
    transformer_layer->infer(infer_param);
  cudaDeviceSynchronize();
  struct timeval ss, ee;
  gettimeofday(&ss, NULL);
  for (int i = 0; i < ite; ++i)
    transformer_layer->infer(infer_param);
  cudaDeviceSynchronize();
  gettimeofday(&ee, NULL);

  printf("transformer costs %.3f ms\n", diffTime(ss, ee) / ite);

  delete transformer_layer;
}
