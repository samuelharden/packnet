#!/bin/bash
# Runs the Learning without Forgetting (LwF) method on given task sequence.
# ./scripts/run_lwf.sh csf ../checkpoints/imagenet/vgg16.pt 2,3

# This is hard-coded to prevent silly mistakes.
declare -A DATASETS
DATASETS["i"]="imdb"
DATASETS["a"]="ag"
DATASETS["o"]="ag12"
DATASETS["t"]="ag34"
declare -A NUM_OUTPUTS
NUM_OUTPUTS["imdb"]="2"
NUM_OUTPUTS["ag"]="4"
NUM_OUTPUTS["ag12"]="2"
NUM_OUTPUTS["ag34"]="2"

ORDER=$1
LOADNAME=$2
GPU_IDS=$3
MODE=$4

for (( i=0; i<${#ORDER}; i++ )); do
  dataset=${DATASETS[${ORDER:$i:1}]}

  mkdir -p ../checkpoints/$dataset/lwf_$ORDER/
  mkdir -p ../logs/$dataset/lwf_$ORDER/

  if [ $i -eq 0 ]
  then
    loadname=$LOADNAME
  else
    loadname=$ft_savename'.pt'
  fi

  tag=$ORDER
  ft_savename=../checkpoints/$dataset/lwf_$ORDER/$tag
  logname=../logs/$dataset/lwf_$ORDER/$tag

  ##############################################################################
  # Train on current dataset.
  ##############################################################################
  CUDA_VISIBLE_DEVICES=$GPU_IDS python lwf.py --mode $MODE \
    --dataset $dataset --num_outputs ${NUM_OUTPUTS[$dataset]} \
    --train_path ../data/$dataset/\
    --test_path ../data/$dataset \
    --loadname $loadname \
    --lr 1e-3 --lr_decay_every 10 --lr_decay_factor 0.1 --finetune_epochs 20 \
    --save_prefix $ft_savename | tee $logname'.txt'
done
