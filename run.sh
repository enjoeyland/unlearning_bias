#!/bin/bash
#SBATCH -J UnlearningBias
#SBATCH --gres=gpu:2
#SBATCH --output=outputs/output_%j.out
#SBATCH --time 4:00:00
gpustat
python -u run.py