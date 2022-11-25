#!/bin/bash

python main.py  \
--data_path_source data/datasets/office31/  \
--src amazon  \
--data_path_target data/datasets/office31/ \
--tar webcam \
--data_path_target_t data/datasets/office31/ \
--tar_t webcam \
--workers 1 \
--pretrained_path checkpoints/amazon_to_webcam_resnet50.pkl \
--batch_size 32 \
--pretrained \
--epochs 200 \
--cluster_iter 100 \
--lr 0.001 \
--domain_adv \
--dis_src \
--dis_tar \
--dis_feat_src \
--dis_feat_tar \
--conf_pseudo_label \
--tsne
