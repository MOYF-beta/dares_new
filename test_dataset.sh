#!/usr/bin/bash

#$ -wd /cluster/project7/Llava_2024/changhao/dares_new
#$ -l tmem=32G
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N test_exp_setup_local
#$ -o /cluster/project7/Llava_2024/changhao/dares_new/test_exp_setup_local.log
#$ -e /cluster/project7/Llava_2024/changhao/dares_new/test_exp_setup_local.err

export PATH=/share/apps/cuda-11.8/bin:/usr/local/cuda-11.8/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LD_LIBRARY_PATH}
export CUDA_INC_DIR=/share/apps/cuda-11.8/include
export LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LIBRARY_PATH}

source /cluster/project7/Llava_2024/venvs/anaconda3/etc/profile.d/conda.sh
conda activate monodepth
python exps/exp_setup_local.py
