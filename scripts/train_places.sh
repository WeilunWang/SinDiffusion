#!/bin/bash
export NCCL_P2P_DISABLE=1

day=$(date "+%Y%m%d")
commit_seq=$(git rev-parse --short HEAD)

for ((i=1;i<=50;i++))

do

# Training
export LOGDIR="OUTPUT/OUTPUT-Places/sinddpm-place$i-$day-$commit_seq"

mpiexec -n 8 python image_train.py --data_dir data/places50/$i.png --diffusion_steps 1000 --image_size 256 \
    --noise_schedule linear --num_channels 64 --num_head_channels 16 --num_res_blocks 1 --channel_mult "1,2,4" \
    --attention_resolution "2" --resblock_updown False --use_fp16 True --use_scale_shift_norm True  --use_checkpoint True

done