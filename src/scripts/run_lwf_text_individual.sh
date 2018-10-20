#!/bin/bash
# Runs the Learning without Forgetting (LwF) method on given task sequence.
# ./scripts/run_lwf.sh csf ../checkpoints/imagenet/vgg16.pt 2,3 finetune imdb
# ./scripts/run_lwf.sh csf ../checkpoints/imagenet/vgg16.pt 2,3 eval imdb
# ./scripts/run_lwf.sh csf ../checkpoints/imagenet/vgg16.pt 2,3 finetune ag
# ./scripts/run_lwf.sh csf ../checkpoints/imagenet/vgg16.pt 2,3 eval ag

# This is hard-coded to prevent silly mistakes.
declare -A DATASETS
DATASETS["j"]="imdb2"
DATASETS["i"]="imdb"
DATASETS["a"]="ag"
declare -A NUM_OUTPUTS
NUM_OUTPUTS["imdb2"]="2"
NUM_OUTPUTS["imdb"]="2"
NUM_OUTPUTS["ag"]="4"

ORDER=$1
LOADNAME=$2
GPU_IDS=$3
MODE=$4
dataset=$5
FTNAME=$6
tag=$ORDER
if [ -f $LOADNAME ]
then
  ft_savename=$FTNAME
  loadname=$LOADNAME
  logname=../logs/$tag
else
  mkdir -p ../checkpoints/$dataset/lwf_$ORDER/
  mkdir -p ../logs/$dataset/lwf_$ORDER/
  ft_savename=../checkpoints/$dataset/lwf_$ORDER/$tag
  logname=../logs/$dataset/lwf_$ORDER/$tag
  loadname=$ft_savename'.pt'
fi


  ##############################################################################
  # Train on current dataset.
  ##############################################################################
  CUDA_VISIBLE_DEVICES=$GPU_IDS python lwf.py --mode $MODE \
    --dataset $dataset --num_outputs ${NUM_OUTPUTS[$dataset]} \
    --train_path ../data/$dataset/\
    --test_path ../data/$dataset \
    --loadname $loadname \
    --lr 1e-3 --lr_decay_every 10 --lr_decay_factor 0.1 --finetune_epochs 40 \
    --save_prefix $ft_savename | tee $logname'.txt'
