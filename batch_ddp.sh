#!/bin/bash
# ============================================================
# Entrenamiento DDP — multi-GPU, nodo único
#
# Uso:
#   2x H200:
#     sbatch -p P2 -w c04 \
#            --ntasks=1 --gpus-per-task=2 \
#            --cpus-per-task=96 --mem=500G \
#            batch_ddp.sh
#
#   4x H100:
#     sbatch -p P2 -w c03 \
#            --ntasks=1 --gpus-per-task=4 \
#            --cpus-per-task=128 --mem=500G \
#            batch_ddp.sh
#
# Nota sobre CPUs:
#   Cada proceso DALI usa recommended_num_threads hilos de CPU.
#     2x H200 → 2 × 48 = 96 CPUs
#     4x H100 → 4 × 32 = 128 CPUs
# ============================================================
set -euo pipefail

echo "===================================="
echo "JobID: ${SLURM_JOB_ID:-no_slurm}"
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "CWD:   $(pwd)"
echo "GPUs:  ${SLURM_GPUS_PER_TASK:-?}"
echo "===================================="

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOGDIR="$SUBMIT_DIR/logs"
mkdir -p "$LOGDIR"

exec > >(tee -a "$LOGDIR/run_ddp_${SLURM_JOB_ID:-local}.out") 2> >(tee -a "$LOGDIR/run_ddp_${SLURM_JOB_ID:-local}.err" >&2)

source /home/almeida/Cristobal/venv/bin/activate

cd "$SUBMIT_DIR"

export PYTHONPATH="$SUBMIT_DIR/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/almeida/opt/eml/lib:/home/almeida/opt/confuse/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

echo "PYTHONPATH: $PYTHONPATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

python - <<'PY'
import eml
print("EML devices:", eml.getDevices())
PY

# Número de GPUs asignadas por SLURM (por defecto 2 si la variable no está)
NGPUS=${SLURM_GPUS_PER_TASK:-2}

# Puerto único por trabajo: evita conflictos si hay varios jobs en el mismo nodo
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "Lanzando torchrun: $NGPUS GPUs, puerto $MASTER_PORT"

torchrun \
    --nproc_per_node=$NGPUS \
    --master_addr=$(hostname) \
    --master_port=$MASTER_PORT \
    -m transformer.distributed.train_ddp

echo "===================================="
echo "End: $(date)"
echo "===================================="
