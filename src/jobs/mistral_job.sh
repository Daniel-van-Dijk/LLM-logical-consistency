#!/bin/bash
  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --job-name=mistral
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=output/mistral_%A.out



# Run your code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_venv_final
srun python -u models/mistral7B.py
