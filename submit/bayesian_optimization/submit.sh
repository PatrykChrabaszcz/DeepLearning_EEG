#!/bin/bash
#
#SBATCH -p meta_gpu-x
#SBATCH --gres gpu:1
#SBATCH -D /home/chrabasp/Workspace/EEG
#
#SBATCH -o /home/chrabasp/Workspace/EEG/meta_logs_bo/o_log_%A.txt
#SBATCH -e /home/chrabasp/Workspace/EEG/meta_logs_bo/e_log_%A.txt
#

source /home/chrabasp/Workspace/env/bin/activate
export LD_LIBRARY_PATH=/home/chrabasp/cuda-8.0/lib64/

echo $HOSTNAME
echo $CUDA_VISIBLE_DEVICES
python main.py  --ini_file config/anomaly_simple.ini \
                --experiment_type BayesianOptimization \
                --budget 20 \
                --working_dir /home/chrabasp/EEG_Results/BO_Anomaly \
                --is_master ${IS_MASTER}
