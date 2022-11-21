#!/bin/bash
#SBATCH --job-name=AVATAR
#SBATCH --output=AVATAR_output.txt
#SBATCH --error=AVATAR_error.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpucompute
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

module load cuda11.1/toolkit/11.1.1

# srun python main.py  \
# --data_path_source /data/home/jkataok1/AVATAR2022/data/datasets/office_home/  \
# --src clipart  \
# --data_path_target /data/home/jkataok1/AVATAR2022/data/datasets/office_home/ \
# --tar product \
# --data_path_target_t /data/home/jkataok1/AVATAR2022/data/datasets/office_home/ \
# --tar_t product \
# --workers 1 \
# --pretrained_path /data/home/jkataok1/AVATAR2022/checkpoints/Clipart_to_Product_resnet50.pkl \
# --batch_size 32 \
# --pretrained \
# --epochs 200 \
# --cluster_iter 100 \
# --lr 0.001 \
# --num_classes 65 \
# --domain_adv \
# --dis_src \
# --dis_tar \
# --dis_feat_src \
# --dis_feat_tar \
# --conf_pseudo_label

# srun python main.py  \
# --data_path_source /data/home/jkataok1/AVATAR2022/data/datasets/office_home/  \
# --src clipart  \
# --data_path_target /data/home/jkataok1/AVATAR2022/data/datasets/office_home/ \
# --tar product \
# --data_path_target_t /data/home/jkataok1/AVATAR2022/data/datasets/office_home/ \
# --tar_t product \
# --workers 1 \
# --pretrained_path /data/home/jkataok1/AVATAR2022/checkpoints/Clipart_to_Product_resnet50.pkl \
# --batch_size 32 \
# --pretrained \
# --epochs 200 \
# --cluster_iter 100 \
# --lr 0.001 \
# --num_classes 65 \
# --domain_adv \
# --dis_src \
# --dis_tar \
# --dis_feat_src \
# --dis_feat_tar

# srun python main.py  \
# --data_path_source /data/home/jkataok1/AVATAR2022/data/datasets/office_home/  \
# --src clipart  \
# --data_path_target /data/home/jkataok1/AVATAR2022/data/datasets/office_home/ \
# --tar product \
# --data_path_target_t /data/home/jkataok1/AVATAR2022/data/datasets/office_home/ \
# --tar_t product \
# --workers 1 \
# --pretrained_path /data/home/jkataok1/AVATAR2022/checkpoints/Clipart_to_Product_resnet50.pkl \
# --batch_size 32 \
# --pretrained \
# --epochs 200 \
# --cluster_iter 100 \
# --lr 0.001 \
# --num_classes 65 \
# --domain_adv \
# --dis_src \
# --dis_tar \
# --dis_feat_src

# srun python main.py  \
# --data_path_source /data/home/jkataok1/AVATAR2022/data/datasets/office_home/  \
# --src clipart  \
# --data_path_target /data/home/jkataok1/AVATAR2022/data/datasets/office_home/ \
# --tar product \
# --data_path_target_t /data/home/jkataok1/AVATAR2022/data/datasets/office_home/ \
# --tar_t product \
# --workers 1 \
# --pretrained_path /data/home/jkataok1/AVATAR2022/checkpoints/Clipart_to_Product_resnet50.pkl \
# --batch_size 32 \
# --pretrained \
# --epochs 200 \
# --cluster_iter 100 \
# --lr 0.001 \
# --num_classes 65 \
# --domain_adv \
# --dis_src \
# --dis_tar

# srun python main.py  \
# --data_path_source /data/home/jkataok1/AVATAR2022/data/datasets/office_home/  \
# --src clipart  \
# --data_path_target /data/home/jkataok1/AVATAR2022/data/datasets/office_home/ \
# --tar product \
# --data_path_target_t /data/home/jkataok1/AVATAR2022/data/datasets/office_home/ \
# --tar_t product \
# --workers 1 \
# --pretrained_path /data/home/jkataok1/AVATAR2022/checkpoints/Clipart_to_Product_resnet50.pkl \
# --batch_size 32 \
# --pretrained \
# --epochs 200 \
# --cluster_iter 100 \
# --lr 0.001 \
# --num_classes 65 \
# --domain_adv \
# --dis_src

srun python main.py  \
--data_path_source /data/home/jkataok1/AVATAR2022/data/datasets/office_home/  \
--src clipart  \
--data_path_target /data/home/jkataok1/AVATAR2022/data/datasets/office_home/ \
--tar product \
--data_path_target_t /data/home/jkataok1/AVATAR2022/data/datasets/office_home/ \
--tar_t product \
--workers 1 \
--pretrained_path /data/home/jkataok1/AVATAR2022/checkpoints/Clipart_to_Product_resnet50.pkl \
--batch_size 32 \
--pretrained \
--epochs 200 \
--cluster_iter 100 \
--lr 0.001 \
--num_classes 65 \
--domain_adv