#!/bin/bash
#SBATCH --job-name=AVATAR_OH
#SBATCH --output=AVATAR_OH_output.txt
#SBATCH --error=AVATAR_OH_error.log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpucompute
#SBATCH --mem=60GB
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

if [[ "$7" != "" ]]; then
    SRC_SUBSET=False
else
    SRC_SUBSET=True
fi

if [[ "$8" != "" ]]; then
    TAR_SUBSET=False
else
    TAR_SUBSET=True
fi

if [[ "$9" != "" ]]; then
    MAJORITY_CLASS=False
else
    MAJORITY_CLASS=True
fi

if [[ "$10" != "" ]]; then
    MINORITY_CLASS_RATIO=0
else
    MINORITY_CLASS_RATIO=$10
fi

FILE=$( echo ${DATA##*/} )
MODELPATH="/data/home/jkataok1/alexnet_resnet_finetune/checkpoints/${SRC}_to_${TAR}_${ARCH}_${FILE}.pkl" 

if [[ "$12" != "slurm" ]]; then

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
    --lr 0.001 \
    --num_classes $NCLASS \
    --domain_adv \
    --dis_src \
    --dis_tar \
    --conf_pseudo_label \
    --log ./checkpoints/$FILE


else
    module load cuda11.3/toolkit/11.3.0                                         │
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
    --lr 0.001 \
    --num_classes $NCLASS \
    --domain_adv \
    --dis_src \
    --dis_tar \
    --conf_pseudo_label \
    --log ./checkpoints/$FILE \
    --src_subset $SRC_SUBSET \
    --tar_subset $TAR_SUBSET \
    --majority_class $MAJORITY_CLASS \
    --minority_class_ratio $MINORITY_CLASS_RATIO \
    --ID 1


fi
