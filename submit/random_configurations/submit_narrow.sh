#!/bin/bash
#
#SBATCH -p meta_gpu-x
#SBATCH --gres gpu:1
#SBATCH -D /home/chrabasp/Workspace/EEG
#
#SBATCH -o /home/chrabasp/Workspace/EEG/meta_logs_random/o_log_%A.txt
#SBATCH -e /home/chrabasp/Workspace/EEG/meta_logs_random/e_log_%A.txt
#

source /home/chrabasp/Workspace/env/bin/activate
export LD_LIBRARY_PATH=/home/chrabasp/cuda-8.0/lib64/

python main.py  --ini_file config/anomaly_simple.ini \
                --experiment_type RandomConfiguration \
                --budget 20 \
                --config_file config/anomaly_simple_narrow.pcs \
                --working_dir /home/chrabasp/EEG_Results/RandomConfigurations_TitanX/${JOB_NUM}_narrowB
