#!/bin/bash
#SBATCH -J UnlearningBias
#SBATCH -p suma_a6000
#SBATCH --gres=gpu:2
#SBATCH --output=outputs/output_%j.log
#SBATCH --time 12:00:00

start_date=`date +%Y-%m-%d`
start_time=`date +%H-%M-%S`
echo "Job started at $start_date $start_time"

gpustat
python -u run.py \
    -m \
    logging.progress_bar=tqdm \
    logging.progress_bar_refresh_rate=40 \
    experiment=tabular_with_generation \
    prompt=finetune_zero_shot \
    method.fit_target=false_pred_0 \
    task.data_path.train=data/adult_train_false_pred_0_label_flip.json \
    task.data_path.valid=data/adult_train_false_pred_0_label_flip.json \
    callbacks.eval_steps=1.0 \
    # training.world_size=1 \
    # training.per_device_batch_size=1 \
    # training.gradient_accumulation_steps=128 \


    # experiment=tabular_with_generation \
    # prompt=finetune_zero_shot \
    # method.fit_target=true_pred \
    # task.data_path.train=data/adult_train_true_pred.json \
    # task.data_path.valid=data/adult_train_true_pred.json \
    # callbacks.eval_steps=0.2 \
    # training.world_size=1 \
    # training.per_device_batch_size=1 \
    # training.gradient_accumulation_steps=128

    # experiment=tabular_with_generation \
    # prompt=find_prediction_dataset \
    # prompt=finetune_zero_shot \

    # training.world_size=1 \
    # training.per_device_batch_size=1 \
    # training.gradient_accumulation_steps=128 \
    # # load_from_checkpoint='.checkpoints_scratch2/llama3.1-8b-instruct/adult/finetune_zero_shot/BS128_Pfinetune_zero_shot_S42/prompt_acc\=0.861-bal_acc\=0.785-eo\=0.1828-spd\=0.2102.ckpt' \
    # do_train=false \
    # do_eval=true \
    
    # experiment=tabular_with_generation \
    # prompt=negtaskvector_zero_shot \
    # method.trained_models.false_pred_0.scaling_coef=0,0.2,0.4,0.6,0.8,1 \
    # method.trained_models.false_pred_1.scaling_coef=0,0.2,0.4,0.6,0.8,1 \
    # method.trained_models.true_pred.scaling_coef=1 \


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
end_date=`date +%Y-%m-%d`
end_time=`date +%H-%M-%S`
echo "Job finished at $end_date $end_time"
mv ./outputs/output_$SLURM_JOB_ID.log ./outputs/${SLURM_JOB_ID}_${start_date}_${start_time}_output.log
