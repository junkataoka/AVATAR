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
--cluster_iter 10 \
--lr 0.001 \

# --src_soft_select \
# --embed_softmax \
# --src_cen_first \
# --init_cen_on_st \
