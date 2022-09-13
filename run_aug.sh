#!/bin/bash
#SBATCH --job-name=aug_flow
#SBATCH --time=10:00:00 # hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --partition=gpu #batch on gpus
#SBATCH --mem-per-cpu=1000
#SBATCH --gres="gpu:TeslaA100:1|gpu:GeForceRTX3090:1"

# working directory
WD="$SLURM_SUBMIT_DIR"
cd $WD

# Load needed module
module load PyTorch/1.10.0-fosscuda-2020b

python3 train_aug_flow.py > log_"$SLURM_JOB_NAME"_"$SLURM_JOB_ID".out 2>&1
