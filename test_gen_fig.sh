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

module load cuda11.1/toolkit/11.1.1

srun python main.py  \
--data_path_source /data/home/jkataok1/CycleGAN-PyTorch/data/office31/  \
--src amazon  \
--data_path_target /data/home/jkataok1/CycleGAN-PyTorch/data/office31/ \
--tar webcam \
--data_path_target_t /data/home/jkataok1/CycleGAN-PyTorch/data/office31/ \
--tar_t webcam \
--workers 1 \
--pretrained_path /data/home/jkataok1/alexnet_resnet_finetune/checkpoints/amazon_to_webcam_resnet50.pkl \
--learn_embed \
--src_cls \
--batch_size 32 \
--beta 1.0 \
--pretrained \
--cluster_method kernel_kmeans \
--epochs 2 \
--cluster_iter 100 \
--lr 0.001 \
--src_soft_select \
--init_cen_on_st \
--resume /data/home/jkataok1/SRDC_CVPR2020/checkpoints/office31_adapt_amazon2webcam_bs32_resnet50_lr0.001_kernel_kmeans/checkpoint.pth.tar
