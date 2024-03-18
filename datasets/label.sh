#!/bin/bash
#
#SBATCH --job-name=LLaVaMatter
#
#SBATCH --output=output_attn.txt
#SBATCH --error=error_attn.txt
#
#SBATCH --partition=gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#
##here your script begins by loading your own bash defaults:
# shellcheck disable=SC1090
source ~/.bashrc

## just if conda is used
#eval "$(conda shell.bash hook)"
#conda activate minko3d

## print additional info
# shellcheck disable=SC2046
echo .... Running on $(hostname) ....
echo available cuda devices have IDs: $CUDA_VISIBLE_DEVICES


python llava_label_matterport.py
