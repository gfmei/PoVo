#!/bin/bash
#
#SBATCH --job-name=PoVo48
#
#SBATCH --output=llava_extr_output.txt
#SBATCH --error=llava_extr_error.txt
#
#SBATCH --partition=cliffjumper
#SBATCH --account=cliffjumper
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16

source /storage3/TEV/gmei/Anaconda3/bin/activate

echo .... Running on $(hostname) ....
echo available cuda devices have IDs: $CUDA_VISIBLE_DEVICES


python llava_label_matterport.py