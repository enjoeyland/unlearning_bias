#!/bin/bash
#SBATCH -J UnlearningBias
#SBATCH -p suma_a6000
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/output_%j.log
#SBATCH --time 6:00:00
date=`date +%Y-%m-%d`
time=`date +%H-%M-%S`

gpustat
python -u run.py \
    -m \
    logging.progress_bar=tqdm \
    logging.progress_bar_refresh_rate=40 \
    experiment=tabular_with_generation \
    model=gpt-3.5

    # model=llama3-8b \
    # method.fit_target=retain \
    # model.learning_rate=3e-5\

    # training.use_lora=false \
    # training.dp_strategy=deepspeed_stage_3 \
    # training.world_size=2 \
    # training.per_device_batch_size=1 \
    # training.gradient_accumulation_steps=8 \

    # training.seed=0 \

    ### negative task vector
    # method=negtaskvector_tabular \
    # method.retain_scaling_coef=0.9 \
    # method.forget_scaling_coef=0,0.4,0.8,1.2,1.6,2,2.4,2.8,3.2,3.6,4,4.4,4.8,5.2,5.6,6 \

    ### shuffled not working..
    # method.load_ckpts.forget='forget_acc=1.000-eo=1.0000-spd=1.0000-v1.ckpt' \
    # method.load_ckpts.retain='retain_acc=0.874-eo=0.5616-spd=0.2091.ckpt' \
    
    ### compas
    # task=compas \
    # method.fit_target=retain,forget \
  

    # experiment=dige \
    # callbacks.max_tolerance=3 \
    # callbacks.early_stop_step=null
    
    # training.world_size=2 \
    # training.per_device_batch_size=2 \
    # training.gradient_accumulation_steps=4 \

    # task=crows_pairs \
    # model=opt-6.7b \

mv ./outputs/output_$SLURM_JOB_ID.log ./outputs/${SLURM_JOB_ID}_${date}_${time}_output.log