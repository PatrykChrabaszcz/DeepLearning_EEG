#!/bin/bash

for i in {1..5}
do
    echo "Submitting random job"
    #sbatch --export=JOB_NUM=${i} submit.sh
    sbatch --export=JOB_NUM=${i} submit_narrow.sh
done