#!/bin/bash

python main.py --multirun hydra/launcher=gpu_slurm \
    dataset=handwriting_single_day \
    dataset.sessions=['t5.2019.05.08'],['t5.2019.11.25'],['t5.2019.12.09'],['t5.2019.12.11'],['t5.2019.12.18'],['t5.2019.12.20'],['t5.2020.01.06'],['t5.2020.01.08'],['t5.2020.01.13'],['t5.2020.01.15'] \
    seed=0,100,1000 \
    batchSize=32 \
    outputDir=/scratch/users/stfan/handwriting_logs/single_days_multi_seeds/