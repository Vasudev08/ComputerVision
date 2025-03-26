#!/bin/bash
#SBATCH --job-name=cv-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=output_%j.log

echo "Loading Anaconda3 ..." && module load Anaconda3
echo "Anaconda Loaded."

echo "Changing to project directory .."
cd /scratch/user/vasu14devagarwal/ComputerVision

echo "Starting evaluating with Conda..."
conda run -n cv-env python evaluation.py

echo "Job Finished!"
