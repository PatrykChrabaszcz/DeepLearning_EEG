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

echo 'HostName' $HOSTNAME
echo 'Device' $CUDA_VISIBLE_DEVICES

# Some problems with this particular GPU, Could happen that we do not start master (but unlikely)!!
if [ "$HOSTNAME" != "metagpui" ] || [ "$CUDA_VISIBLE_DEVICES" != "2" ]
then
echo OKEY
python main.py  --ini_file config/anomaly_simple.ini \
                --experiment_type BayesianOptimization \
                --budget 20 \
                --working_dir /home/chrabasp/EEG_Results/BO_Anomaly_6 \
                --is_master ${IS_MASTER}
else
sleep 30
fi