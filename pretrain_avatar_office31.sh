#!/bin/bash
#SBATCH --job-name=AVATAR_pretrain
#SBATCH --output=AVATAR.txt
#SBATCH --error=AVATAR_error.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpucompute
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpu:2

SRC="$1" # source domain
TAR="$2" # target domain
source ~/mlenv/bin/activate
module load cuda11.1/toolkit/11.1.1
srun python src/main.py --src_data=office31 --tar_data=office31 --src_domain=$SRC --tar_domain=$TAR --lr=0.005 --batch_size=32 --epochs=200

#srun python src/main.py --src_domain=$SRC --tar_domain=$TAR --lr=0.005 --batch_size=128 --epochs=600 --pretrained \
#        --pretrained_path=$MODELPATH
#srun python tests/src/test_cuda.py




