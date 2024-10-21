#!/bin/bash
#SBATCH -J UnlearningBias
#SBATCH --gres=gpu:2
#SBATCH --output=outputs/output_%j.out
#SBATCH --time 6:00:00
gpustat

python -u run.py \
    -m \
    experiment=dige \
    logging.progress_bar=tqdm \
    logging.progress_bar_refresh_rate=40 \
    callbacks.max_tolerance=3 \
    callbacks.early_stop_step=null
    # method=negtaskvector_adult \
    # method.retain_scaling_coef=0.9 \
    # method.forget_scaling_coef=0,0.4,0.8,1.2,1.6,2,2.4
    # experiment=dige \
    # do_train=false \
    # do_test=true \
    # training.world_size=2 \
    # training.per_device_batch_size=2 \
    # training.gradient_accumulation_steps=4 \
    # task=crows_pairs \
    # model=opt-6.7b \
