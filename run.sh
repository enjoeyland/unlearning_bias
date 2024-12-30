#!/bin/bash
#SBATCH -J UnlearningBias
#SBATCH --gres=gpu:2
#SBATCH --output=outputs/output_%j.log
#SBATCH --time 6:00:00
date=`date +%Y-%m-%d`
time=`date +%H-%M-%S`

gpustat


python -u run.py \
    -m \
    logging.progress_bar=tqdm \
    logging.progress_bar_refresh_rate=40 \
    method=regularization\
    method.regularization_weight=0.1,0.5,1
    
    ### negative task vector
    # method=negtaskvector_tabular \
    # method.forget_scaling_coef=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
    # method.keywords=[remove]\
    # task=adult_filter_remove\
    # method.make_perpendicular=true 

    # training.use_lora=false \
    # training.dp_strategy=deepspeed_stage_3 \
    # training.world_size=2 \
    # training.per_device_batch_size=1 \
    # training.gradient_accumulation_steps=8 \

    # training.seed=0 \

    

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


mkdir ./outputs/${date}
mkdir ./outputs/${date}/${time}
mv ./outputs/output_$SLURM_JOB_ID.log ./outputs/${date}/${time}/output.log