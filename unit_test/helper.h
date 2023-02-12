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
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <sys/time.h>

using namespace std;

double diffTime(timeval start, timeval end) {
  return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

template <typename T>
void device_malloc(T **ptr, int size) {
  cudaMalloc((void **)ptr, sizeof(T) * size);
}

template <typename T>
void load_weights(T **ptr, char *file, int size) {
  FILE *fd = fopen(file, "r");
  if (fd == NULL) {
    printf("FILE %s does not exist.\n", file);
    // exit(0);
    return;
  }

  T *h_ptr = (T *)malloc(sizeof(T) * size);
  for (int i = 0; i < size; ++i) {
    float tmp;
    fscanf(fd, "%f", &tmp);
    h_ptr[i] = (T)tmp;
  }
  cudaMemcpy(*ptr, h_ptr, sizeof(T) * size, cudaMemcpyHostToDevice);
  free(h_ptr);
}

template <typename T>
void result_check(const T *ptr, char *file, int size) {
  T *h_ptr = (T *)malloc(sizeof(T) * size);
  cudaMemcpy(h_ptr, ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);

  float max_diff = -1e20f;
  FILE *fd = fopen(file, "r");
  if (fd == NULL) {
    printf("FILE %s does not exist.\n", file);
    // exit(0);
    return;
  }
  for (int i = 0; i < size; ++i) {
    float tmp;
    fscanf(fd, "%f", &tmp);
    float diff = fabs(tmp - (float)h_ptr[i]);
    if (diff > max_diff) {
      max_diff = diff;
      printf("%d CUDA %f Expect %f diff %f\n", i, (float)h_ptr[i], (float)tmp, (float)diff);
    }
  }
  free(h_ptr);
  fclose(fd);

  printf("\033[0;32m");
  printf("Check with %s max_diff %f\n", file, max_diff);
  printf("\033[0m");
}

template <typename T>
void print_vec(const T *data, const char *str, const int size) {
  printf("print %s\n", str);
  T *tmp = (T *)malloc(sizeof(T) * size);
  cudaMemcpy(tmp, data, sizeof(T) * size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < size; ++i)
    printf("%d %f\n", i, (float)tmp[i]);
  free(tmp);
  printf("\n");
}
