#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

method="negtaskvector"

task="bmlama53"
langs=("en" "fr" "es" "pt" "ar" "vi" "ca" "hi" "bn")
max_length=32

model_name="bloom-3b"
world_size=2
batch_size=8
warmup_ratio=0
dp_strategy="deepspeed_stage_2"


seed=("42")
learning_rate=("1e-4" "3e-4" "5e-5")
fit_target=("retain")

for s in "${seed[@]}"; do
for lr in "${learning_rate[@]}"; do
for ft in "${fit_target[@]}"; do
echo "Running $method $task $s $lr $ft"
python run.py \
    --model_name $model_name \
    --model "bigscience/$model_name" \
    --method $method \
    --cache_dir ../.cache \
    --task $task \
    --forget_lang ${langs[@]} \
    --retain_lang ${langs[@]} \
    --forget_num 32 \
    --retain_multiplier 1 \
    --max_length $max_length \
    --num_workers 4 \
    --data_dir ../../research/multilingual-unlearning/data/ \
    --fit_target $ft \
    --do_train \
    --seed $s \
    --dp_strategy $dp_strategy \
    --bf16 \
    --optimizer adamw \
    --learning_rate $lr \
    --lr_scheduler_type linear \
    --warmup_ratio $warmup_ratio \
    --epochs 30 \
    --world_size $world_size \
    --per_device_batch_size $batch_size \
    --gradient_accumulation_steps $((32 / batch_size / world_size)) \
    --logging_steps 32 \
    --eval_steps 1 \
    --max_tolerance 30 \
    --output_dir ".checkpoints/"
done
done
done