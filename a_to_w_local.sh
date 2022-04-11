#!/bin/bash
#SBATCH --job-name=SRDC
#SBATCH --output=SRDC_output.txt
#SBATCH --error=SRDC_error.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpucompute
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

# module load cuda11.1/toolkit/11.1.1

python main.py  \
--data_path_source /home/jun/GoogleDrive/project/dataset/office31 \
--src amazon  \
--data_path_target /home/jun/GoogleDrive/project/dataset/office31 \
--tar webcam \
--data_path_target_t /home/jun/GoogleDrive/project/dataset/office31 \
--tar_t webcam \
--workers 1 \
--pretrained \
--pretrained_path /home/jun/GoogleDrive/project/models/amazon_to_webcam_resnet50.pkl \
--learn_embed \
--src_cls \
--batch_size 24 \
--beta 1.0 \
--pretrained \
--cluster_method kernel_kmeans \
--epochs 200 \
--cluster_iter 20 \
--lr 0.001 \
--src_soft_select \
--stop_epoch 20 \
--init_cen_on_st \

# python main.py  \
# --data_path_source /home/jun/GoogleDrive/project/dataset/office31 \
# --src dslr  \
# --data_path_target /home/jun/GoogleDrive/project/dataset/office31 \
# --tar webcam \
# --data_path_target_t /home/jun/GoogleDrive/project/dataset/office31 \
# --tar_t webcam \
# --workers 1 \
# --pretrained \
# --pretrained_path /home/jun/GoogleDrive/project/models/dslr_to_webcam_resnet50.pkl \
# --learn_embed \
# --src_cls \
# --batch_size 24 \
# --beta 1.0 \
# --pretrained \
# --cluster_method kernel_kmeans \
# --epochs 200 \
# --cluster_iter 20 \
# --lr 0.001 \
# --src_soft_select \
# --init_cen_on_st \
# --stop_epoch 20

# python main.py  \
# --data_path_source /home/jun/GoogleDrive/project/dataset/office31 \
# --src webcam  \
# --data_path_target /home/jun/GoogleDrive/project/dataset/office31 \
# --tar dslr \
# --data_path_target_t /home/jun/GoogleDrive/project/dataset/office31 \
# --tar_t dslr \
# --workers 1 \
# --pretrained \
# --pretrained_path /home/jun/GoogleDrive/project/models/webcam_to_dslr_resnet50.pkl \
# --learn_embed \
# --src_cls \
# --batch_size 24 \
# --beta 1.0 \
# --pretrained \
# --cluster_method kernel_kmeans \
# --epochs 200 \
# --cluster_iter 20 \
# --lr 0.001 \
# --src_soft_select \
# --init_cen_on_st \
# --stop_epoch 20


# python main.py  \
# --data_path_source /home/jun/GoogleDrive/project/dataset/office31 \
# --src amazon  \
# --data_path_target /home/jun/GoogleDrive/project/dataset/office31 \
# --tar dslr \
# --data_path_target_t /home/jun/GoogleDrive/project/dataset/office31 \
# --tar_t dslr \
# --workers 1 \
# --pretrained \
# --pretrained_path /home/jun/GoogleDrive/project/models/amazon_to_dslr_resnet50.pkl \
# --learn_embed \
# --src_cls \
# --batch_size 24 \
# --beta 1.0 \
# --pretrained \
# --cluster_method kernel_kmeans \
# --epochs 200 \
# --cluster_iter 100 \
# --lr 0.001 \
# --src_soft_select \
# --init_cen_on_st \
# --stop_epoch 10

# python main.py  \
# --data_path_source /home/jun/GoogleDrive/project/dataset/office31 \
# --src dslr  \
# --data_path_target /home/jun/GoogleDrive/project/dataset/office31 \
# --tar amazon \
# --data_path_target_t /home/jun/GoogleDrive/project/dataset/office31 \
# --tar_t amazpn \
# --workers 1 \
# --pretrained \
# --pretrained_path /home/jun/GoogleDrive/project/models/dslr_to_amazon_resnet50.pkl \
# --learn_embed \
# --src_cls \
# --batch_size 24 \
# --beta 1.0 \
# --pretrained \
# --cluster_method kernel_kmeans \
# --epochs 200 \
# --cluster_iter 20 \
# --lr 0.001 \
# --src_soft_select \
# --init_cen_on_st \
# --stop_epoch 10

# python main.py  \
# --data_path_source /home/jun/GoogleDrive/project/dataset/office31 \
# --src webcam  \
# --data_path_target /home/jun/GoogleDrive/project/dataset/office31 \
# --tar amazon \
# --data_path_target_t /home/jun/GoogleDrive/project/dataset/office31 \
# --tar_t amazon \
# --workers 1 \
# --pretrained \
# --pretrained_path /home/jun/GoogleDrive/project/models/webcam_to_amazon_resnet50.pkl \
# --learn_embed \
# --src_cls \
# --batch_size 24 \
# --beta 1.0 \
# --pretrained \
# --cluster_method kernel_kmeans \
# --epochs 200 \
# --cluster_iter 20 \
# --lr 0.001 \
# --src_soft_select \
# --init_cen_on_st \
# --stop_epoch 20