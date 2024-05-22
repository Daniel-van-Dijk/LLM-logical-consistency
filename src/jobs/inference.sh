#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=train_model
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:01:00
#SBATCH --output=slurm/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate ACTS-practical-1
srun python -u src/inference.py --model mistral7B --run_tasks temporal