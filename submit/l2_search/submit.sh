#!/bin/bash
#
#SBATCH -p meta_gpu-x
#SBATCH --gres gpu:1
#SBATCH -D /home/chrabasp/Workspace/EEG
#
#SBATCH -o /home/chrabasp/Workspace/EEG/meta_logs/o_log_%A.txt
#SBATCH -e /home/chrabasp/Workspace/EEG/meta_logs/e_log_%A.txt
#

source /home/chrabasp/Workspace/env/bin/activate
export LD_LIBRARY_PATH=/home/chrabasp/cuda-8.0/lib64/
echo "l2_decay ${L2_DECAY}"
echo "rnn_hidden_size ${RNN_HIDDEN_SIZE}"
echo "optimizer ${OPTIMIZER}"
python main.py  --ini_file config/anomaly_simple.ini \
                --experiment_type SingleTrain \
                --optimizer AdamW \
                --l2_decay ${L2_DECAY} \
                --rnn_hidden_size ${RNN_HIDDEN_SIZE} \
                --optimizer ${OPTIMIZER} \
                --budget 60 \
                --working_dir /home/chrabasp/EEG_Results/L2_Test
