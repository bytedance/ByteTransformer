#!/bin/bash
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

n_layers=12
precisions=("fp16")
batch_sizes=(1 2 4 8 16)
seqlens=(32 64 128 192 256 320 384 448 512 576 640 704 768 832 896 960 1024)
head_num=12
head_size=64
avg_seqlen_percent=60
gpu_card=0
export CUDA_VISIBLE_DEVICES=${gpu_card}

for precision in ${precisions[@]}; do
    echo "Presicion is ${precision}"
    logdir="bert-bench-${precision}"
    if [ ! -f ${logdir} ]; then
        mkdir ${logdir} -p
    fi
    cat /proc/cpuinfo >${logdir}/cpuinfo.txt
    nvidia-smi -q -i $gpu_card >${logdir}/gpuinfo.txt

    all_log="${logdir}/all-log.log"
    echo "Writing results to ${all_log}"
    printf "%-15s%-15s%-15s%-15s\n" "Batch Size" "Seqlen" "Precision" "BT Latency (ms)" >$all_log

    for batch_size in ${batch_sizes[@]}; do
        for seqlen in ${seqlens[@]}; do
            tmp_log=${logdir}/layers-${n_layers}-bs-${batch_size}-seq-${seq_len}-${precision}.log
            avg_seqlen=$((seqlen * avg_seqlen_percent / 100))
            echo "processing batch_size ${batch_size} seqlen ${seqlen} avg_seqlen ${avg_seqlen}"

            python bert_transformer_test.py ${batch_size} ${seqlen} ${head_num} ${head_size} --avg_seqlen=${avg_seqlen} --n_layers=${n_layers} --dtype=${precision} 2>&1 | tee $tmp_log

            bt_time=$(tail -n 1 ${tmp_log} | head -n 1 | awk '{print $3}')

            printf "%-15s%-15s%-15s%-15s\n" ${batch_size} ${seqlen} ${precision} ${bt_time} >>$all_log
        done
    done
done
