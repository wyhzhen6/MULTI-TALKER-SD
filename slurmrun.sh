#!/usr/bin/env bash
##SBATCH -o gpu-job-%j.output
#SBATCH --gres=gpu:4
#SBATCH -n 4
# conda activate whisper-env

./run.sh