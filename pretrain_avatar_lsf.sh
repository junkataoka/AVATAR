#!/bin/bash
#BSUB -J avatar_pretrain
#BSUB -cwd /home/kataokaj/scratch/AVATAR
#BSUB -o log-output.txt
#BSUB -e log_error.txt
#BSUB -n 1
#BSUB -q long
#BSUB -n 1
#BSUB -gpu "num=1"

#source /apps/rocs/init.sh

DATA="$1"
SRC="$2" # source domain
TAR="$3" # target domain

WANDB_CONFIG_DIR=/home/kataokaj/scratch/AVATAR/tmp/
WANDB_DIR=/home/kataokaj/scratch/AVATAR/tmp/
WANDB_CACHE_DIR=/home/kataokaj/scratch/AVATAR/tmp/
ml CUDA/11.0.2-GCC-9.3.0 cuDNN/8.0.4.30-CUDA-11.0.2  
# python src/main.py --src_data=$DATA --tar_data=$DATA --src_domain=$SRC --tar_domain=$TAR --lr=0.005 --batch_size=128 --epochs=200
python src/main.py --src_data=office31 --tar_data=office31 --src_domain=amazon --tar_domain=webcam --lr=0.005 --batch_size=128 --epochs=200

#srun python src/main.py --src_domain=$SRC --tar_domain=$TAR --lr=0.005 --batch_size=128 --epochs=600 --pretrained \
#        --pretrained_path=$MODELPATH
#srun python tests/src/test_cuda.py




