#!/bin/bash
#SBATCH --job-name=ddp_h100
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ddp_h100_%j.out
#SBATCH --error=logs/ddp_h100_%j.err

# ── Entorno ─────────────────────────────────────────────────────────────────
source /home/almeida/Cristobal/venv/bin/activate
cd /home/almeida/Cristobal/High-Performance-Vision-Transformer

mkdir -p logs

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "Start       : $(date)"

# ── Entrenamiento DDP ────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29501 \
    -m src.transformer.distributed.train_ddp

echo "End : $(date)"
