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
--src webcam  \
--data_path_target /data/home/jkataok1/CycleGAN-PyTorch/data/office31/ \
--tar amazon \
--data_path_target_t /data/home/jkataok1/CycleGAN-PyTorch/data/office31/ \
--tar_t amazon \
--workers 1 \
--pretrained_path /data/home/jkataok1/alexnet_resnet_finetune/checkpoints/webcam_to_amazon_resnet50.pkl \
--learn_embed \
--src_cls \
--batch_size 42 \
--beta 0.5 \
--pretrained \
--src_soft_select \
--embed_softmax \
--init_cen_on_st \
--cluster_iter 20
# --cluster_method kernel_kmeans
# --src_cen_first \
# --initial_cluster 2 \