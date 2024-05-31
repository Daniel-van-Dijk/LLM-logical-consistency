#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=train_model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=00:01:00
#SBATCH --output=slurm/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate llm_venv_final
srun python -u src/evaluate.py --model llama3_8B --task comparative --prompt_type zero_shot_cot --evaluation_type llm 