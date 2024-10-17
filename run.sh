#!/bin/bash
#SBATCH -J UnlearningBias
#SBATCH --gres=gpu:2
#SBATCH --output=outputs/output_%j.out
#SBATCH --time 6:00:00
gpustat
# python -u run.py logging.progress_bar_refresh_rate=20

# python -u run.py method=original do_train=false do_test=true training.world_size=1 training.per_device_batch_size=1 training.gradient_accumulation_steps=16 task=crows_pairs model=opt-6.7b
# python -u run.py method=original do_train=false do_test=true training.world_size=1 training.per_device_batch_size=1 training.gradient_accumulation_steps=16 task=crows_pairs model=llama2-7b
# python -u run.py -m method.fit_target=retain training.world_size=1 training.per_device_batch_size=1 training.gradient_accumulation_steps=16 model=opt-6.7b,opt-2.7b,llama2-7b
python -u run.py -m method.fit_target=retain,forget logging.progress_bar_refresh_rate=20
# python -u run.py experiment=dige 