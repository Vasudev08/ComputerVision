#!/bin/bash
#SBATCH --job-name=cv-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=output_%j.log

module load Anaconda3
source activate cv-env

cd ~/ComputerVsision
python train.py