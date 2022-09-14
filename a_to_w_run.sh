#!/bin/bash

python main.py  \
--data_path_source /data/home/jkataok1/AVATAR/data/office31/  \
--src amazon  \
--data_path_target /data/home/jkataok1/AVATAR/data/office31/ \
--tar webcam \
--data_path_target_t /data/home/jkataok1/AVATAR/data/office31/ \
--tar_t webcam \
--workers 1 \
--pretrained_path /data/home/jkataok1/AVATAR/checkpoints/amazon_to_webcam_resnet50.pkl \
--batch_size 32 \
--pretrained \
--epochs 200 \
--cluster_iter 100 \
--lr 0.001 \
