#!/bin/bash

python -m neuralDecoder.main \
    dataset=handwriting_all_days \
    batchSize=48 \
    dataset.syntheticMixingRate=0.3 \
    outputDir=/oak/stanford/groups/shenoy/stfan/logs/handwriting_logs/test_all_days_synthetic_mixing_rate
