#!/bin/bash

echo "Submitting master job"
sbatch --export=JOB_NUM=${0},IS_MASTER=1 submit.sh

for i in {1..18}
do
    echo "Submitting worker job"
    sbatch --export=JOB_NUM=${i},IS_MASTER=0 submit.sh
done

