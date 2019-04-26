#!/usr/bin/env bash
python /project/train/train_classifier.py \
--gpu "0"  \
--fold "$1" \
--num_workers 16 \
--network seresnext50 \
--loss_function cce \
--optimizer adam \
--learning_rate 0.0001 \
--batch_size 32 \
--crop_size 160 \
--epochs 100 \
--reduce_lr_patience 5 \
--reduce_lr_rate 0.5 \
--early_stopping 15 \
--preprocessing_function caffe \
--augment 1 \
--alias with_augs