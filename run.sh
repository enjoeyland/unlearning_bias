#!/bin/bash
#SBATCH -J UnlearningBias
#SBATCH --gres=gpu:2
#SBATCH --output=outputs/output_%j.log
#SBATCH --time 6:00:00
start_date=`date +%Y-%m-%d`
start_time=`date +%H-%M-%S`
echo "Job started at $start_date $start_time"

gpustat


python -u run.py \
    -m \
    logging.progress_bar=tqdm \
    logging.progress_bar_refresh_rate=40 \
    training.bf16=false \
    training.world_size=4 \
    training.per_device_batch_size=8 \
    training.gradient_accumulation_steps=1 \


    # 13658/24576 bf16 속도가 빠르다 6:14
    # 13110/24576 fp32 속도가 느리다(tonser core 미사용) window2 15:50 / window1 22:00 / window4 6:30 역시 좋아 
    # 13110/24576 fp32 속도가 조금 빨라짐(tonser core 사용) 10:04

    # experiment=tabular_with_generation \
    # model=gpt-3.5

    ### regularization
    # method=regularization\
    # method.regularization_weight=0.1,0.5,1

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

end_date=`date +%Y-%m-%d`
end_time=`date +%H-%M-%S`
echo "Job finished at $end_date $end_time"
mv ./outputs/output_$SLURM_JOB_ID.log ./outputs/${SLURM_JOB_ID}_${start_date}_${start_time}_output.log