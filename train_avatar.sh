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


if [[ "$1" != "" ]]; then
    ARCH="$1"
else
    ARCH='vits16'
fi

if [[ "$2" != "" ]]; then
    SRC="$2"
else
    SRC='amazon'
fi

if [[ "$3" != "" ]]; then
    TAR="$3"
else
    TAR='webcam'
fi

if [[ "$4" != "" ]]; then
    NCLASS="$4"
else
    NCLASS=31
fi

if [[ "$5" != "" ]]; then
    DATA="$5"
else
    DATA="/home/junkataoka/AVATAR/data/datasets/office31"
fi

if [[ "$6" != "" ]]; then
    EPOCH="$6"
else
    EPOCH=200
fi
if [[ "$7" != "" ]]; then
    BATCH="$7"
else
    BATCH=32
fi

FILE=$( echo ${DATA##*/} )
MODELPATH="/home/junkataoka/alexnet_resnet_finetune/checkpoints/${SRC}_to_${TAR}_${ARCH}_${FILE}.pkl" 

if [[ "$8" != "slurm" ]]; then

    python main.py \
    --arch $ARCH \
    --data_path_source $DATA \
    --src $SRC \
    --data_path_target $DATA \
    --tar $TAR \
    --data_path_target_t $DATA \
    --tar_t $TAR \
    --workers 1 \
    --pretrained_path $MODELPATH \
    --batch_size $BATCH \
    --epochs $EPOCH \
    --cluster_iter 100 \
    --lr 0.0001 \
    --num_classes $NCLASS \
    --domain_adv \
    --dis_src \
    --dis_tar \
    --conf_pseudo_label \
    --log ./checkpoints/$FILE


else
    module load cuda11.1/toolkit/11.1.1
    conda activate cyclegan

    srun -n1 --gpus=1 --exclusive -c1 python main.py \
    --arch $ARCH \
    --data_path_source $DATA \
    --src $SRC \
    --data_path_target $DATA \
    --tar $TAR \
    --data_path_target_t $DATA \
    --tar_t $TAR \
    --workers 1 \
    --pretrained_path $MODELPATH \
    --batch_size $BATCH \
    --epochs $EPOCH \
    --cluster_iter 100 \
    --lr 0.0001 \
    --num_classes $NCLASS \
    --domain_adv \
    --dis_src \
    --dis_tar \
    --conf_pseudo_label \
    --log ./checkpoints/$FILE

fi
