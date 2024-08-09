#!/usr/bin/env bash

#SBATCH --job-name=run_inference
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=node-rtx8-07
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL

start_time=$(date +%s)

# activate your own conda env here (e.g. controlnet here)
CONDA_DIR="/NAS5/speech/user/zhuyixing276/miniconda3"
ENV_NAME=mya
. "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate $ENV_NAME
echo "$(which python3)"

bash run_inference.sh
