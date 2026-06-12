#!/bin/bash

source /home/almeida/Cristobal/venv/bin/activate
cd /home/almeida/Cristobal/High-Performance-Vision-Transformer

export PYTHONPATH=/home/almeida/Cristobal/High-Performance-Vision-Transformer/src:$PYTHONPATH

mkdir -p logs

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "Start  : $(date)"

torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29500 \
    -m src.transformer.distributed.train_ddp

echo "End : $(date)"
