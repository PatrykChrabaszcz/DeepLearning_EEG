#!/bin/bash

for optimizer in Adam AdamW
    do
    for rnn_hidden_size in 32 256
    do
        for l2_decay in 0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0000001
        do
            echo "Submitting job hidden_size ${rnn_hidden_size} , l2_decay ${l2_decay}, optimizer ${optimizer}"
            sbatch --export=L2_DECAY=${l2_decay},RNN_HIDDEN_SIZE=${rnn_hidden_size},OPTIMIZER=${optimizer} submit.sh
        done
    done
done