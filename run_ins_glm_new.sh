#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# nohup CUDA_VISIBLE_DEVICES=0 deepspeed finetuning_lora_sft.py --num_train_epochs 2 --train_batch_size 2 --lora_r 8   > nohup_lora.out 2>&1 &
# DeepSpeed Team
OUTPUT_PATH=./output/0503-zh-ins/
mkdir -p $OUTPUT_PATH

python train_lora_new.py \
    --dataset_path data/alpaca_zh \
    --lora_rank 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir $OUTPUT_PATH \
    &> $OUTPUT_PATH/training.log