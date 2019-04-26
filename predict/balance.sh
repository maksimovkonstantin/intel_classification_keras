#!/usr/bin/env bash
python prediction_balancing.py /project/submits/predict_tensor.pt /project/submits/balanced_tensor.pt \
--min_alpha 0.0000001 \
--start_alpha 0.0001 \
--iterations 10000
