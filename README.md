<div align="center">

# High-Performance Computing & Machine Learning

### Training, Compressing and Profiling Vision Transformers under an Energy Budget

**Bachelor's Thesis 2025–2026 · University of La Laguna**
*Higher School of Engineering and Technology · Bachelor's Degree in Computer Engineering*

**Author:** Cristóbal Jesús Sarmiento Rodríguez

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/PyTorch_Lightning-792EE5?logo=lightning&logoColor=white)
![NVIDIA DALI](https://img.shields.io/badge/NVIDIA_DALI-76B900?logo=nvidia&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-blue)

</div>

---

## Overview

This repository contains the full experimental pipeline of a Bachelor's Thesis that studies
**how to train and compress Vision Transformers efficiently** while measuring their real
**energy consumption** on HPC hardware (NVIDIA **H100 / H200** GPUs, SLURM cluster).

The project trains an image classifier on a **20-class** dataset of birds, insects and
animals, and then explores a wide range of **model-compression techniques** to trade off
accuracy, latency and energy:

- A **Vision Transformer (ViT)** implemented **from scratch** (attention, encoder blocks, MLP, DropPath, EMA…).
- Strong **baselines**: a fine-tuned **ResNet-50** and a **ViT-B/16** from `torchvision`.
- **Quantization** at multiple bit-widths: `int1`, `int2`, `int4`, `int8`, and `16-bit` (FP16 / BF16).
- **Pruning**: structured (attention-head pruning), unstructured (weight pruning) and **ToMe** (Token Merging).
- **Knowledge Distillation** (vanilla and structured).
- High-throughput data loading with **NVIDIA DALI** and **multi-GPU training** via **DDP + `torchrun` + SLURM**.
- **Energy measurement** for both CPU (Intel RAPL) and GPU (NVML through the **EML / `pyeml`** library).

---

## Repository Structure

```
TrabajoFinGrado/
├── src/
│   ├── transformer/                # ViT implemented from scratch
│   │   ├── models/                 # attention, encoder_block, mlp, drop_path, transformer_model
│   │   ├── data/dali/              # DALI datamodule, pipelines, split & validation utils
│   │   ├── training/               # trainer, config, mixup/cutmix, EMA, schedulers, losses…
│   │   │   └── energy/             # RAPL (CPU) + EML/NVML (GPU) energy meters
│   │   ├── distributed/            # DistributedDataParallel (DDP) training
│   │   └── visualization/          # training-curve plotting
│   │
│   ├── RestNet/                    # ResNet-50 baseline (training + plots)
│   ├── ViTB16/                     # torchvision ViT-B/16 baseline
│   │
│   ├── quantization/               # int1 / int2 / int4 / int8 / 16bits / compare_all
│   ├── pruning/
│   │   ├── structured/             # attention-head pruning (+ sweep)
│   │   ├── unstructured/           # weight pruning (+ sweep)
│   │   ├── tome/                   # Token Merging (+ sweep)
│   │   └── common/                 # checkpoint, evaluator, stats, plotters
│   └── distillation/               # knowledge distillation (vanilla + structured)
│
├── scripts/                        # SLURM batch jobs (H100/H200) + replot helper
├── tests/                          # transformer & convolutional test/eval entry points
├── visualization/                  # cross-model accuracy comparison
├── data/                           # dataset (ImageFolder layout — git-ignored)
├── checkpoints/  outputs/  logs/   # generated artifacts (git-ignored)
└── configs/  notebooks/
```

---

## Requirements

### Hardware
- A **CUDA-capable NVIDIA GPU** is strongly recommended (the project targets H100/H200).
  CPU-only execution is possible for small experiments but very slow.
- **Linux** is required for the energy-measurement features (Intel **RAPL** via `/sys/class/powercap`
  and **NVML** through EML). The rest of the code is cross-platform.

### Software
- **Python 3.11**
- **NVIDIA CUDA** drivers and toolkit compatible with your PyTorch build
- **NVIDIA DALI** (matching your CUDA version)
- An optional **SLURM** scheduler for multi-GPU / batch jobs

### Python packages
| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Core deep-learning framework + pretrained baselines |
| `pytorch-lightning` | `LightningDataModule` orchestration |
| `nvidia-dali-cudaXXX` | GPU-accelerated data loading pipelines |
| `pyeml` | EML energy-measurement library (NVML GPU backend) |
| `numpy`, `pandas` | Numerical & tabular processing |
| `matplotlib` | Plots and figures |
| `Pillow` | Image handling |
| `tqdm` | Progress bars |

---

## Installation

```bash
# 1) Clone the repository
git clone <repo-url> TrabajoFinGrado
cd TrabajoFinGrado

# 2) Create and activate a virtual environment (Python 3.11)
python3.11 -m venv venv
source venv/bin/activate          # Linux / macOS
# .\venv\Scripts\Activate.ps1     # Windows PowerShell

# 3) Install PyTorch (pick the build that matches your CUDA version)
#    See https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4) Install NVIDIA DALI (match your CUDA version, e.g. cuda120)
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120

# 5) Install the remaining dependencies
pip install pytorch-lightning numpy pandas matplotlib Pillow tqdm pyeml
```

> **Note on `pyeml` / energy:** the EML library and the Linux RAPL counters are only available
> on Linux + NVIDIA hardware. The code gracefully reports `available: False` when RAPL is not
> present, so the project still runs (just without energy figures) on other platforms.

---

## Dataset

The data loader expects a classic **ImageFolder** layout — one sub-folder per class:

```
data/dataset2/
├── Bald_eagle/        img001.jpg, img002.jpg, ...
├── Bee/               ...
├── Monarch/           ...
├── White_tailed_deer/ ...
└── ...                # 20 classes in total
```

The 20 classes are a mix of birds, insects and animals (e.g. *Bald_eagle, Bee, Monarch,
Northern_cardinal, Painted_lady, White_heron, White_tailed_deer*…).

- Train/validation splitting is handled automatically (`train_ratio = 0.8` by default).
- Corrupt/unreadable images are detected and (optionally) removed by the DALI validator.
- The dataset directory is **git-ignored** — place your images there before training.

> Set the dataset location with the `root_dir` field of the relevant training config
> (e.g. `TransformerTrainingConfig.root_dir`). On the HPC cluster it defaults to a
> pre-resized `dataset_256` folder.

---

## Usage

All commands are run **from the project root** with `src/` on the Python path:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"          # Linux / macOS
# $env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"    # Windows PowerShell
```

Configuration is **code-based** (Python `@dataclass` configs) rather than CLI flags — edit the
corresponding `config.py` to change hyper-parameters, paths and batch sizes. Many settings are
**auto-tuned to the detected GPU** (`auto_tune_for_gpu = True`).

### 1. Train the Vision Transformer (from scratch)
```bash
python -m transformer.training.train_transformer
```

### 2. Multi-GPU training with DDP (single node, 2 GPUs)
```bash
torchrun --nproc_per_node=2 -m transformer.distributed.train_ddp
```

### 3. Train the baselines
```bash
python -m RestNet.training.train_cnn         # ResNet-50
python -m ViTB16.training.training_vit16      # torchvision ViT-B/16
```

### 4. Quantization experiments
```bash
python -m quantization.int8.compare_quantization      # also: int1 / int2 / int4
python -m quantization.16bits.compare_precision        # FP16 vs BF16 vs FP32
python -m quantization.compare_all.compare_all         # aggregate comparison
```

### 5. Pruning experiments
```bash
# Structured (attention-head pruning)
python -m pruning.structured.run_structured
python -m pruning.structured.run_structured_sweep

# Unstructured (weight pruning)
python -m pruning.unstructured.run_unstructured
python -m pruning.unstructured.run_unstructured_sweep

# ToMe — Token Merging
python -m pruning.tome.run_tome
python -m pruning.tome.run_tome_sweep
```

### 6. Knowledge distillation
```bash
python -m distillation.run_distillation
python -m distillation.run_structured_distillation
```

### 7. Visualization & reporting
```bash
python -m visualization.accuracy_models.compare_models   # cross-model accuracy comparison
python scripts/replot_from_log.py                        # rebuild plots from a training log
```

---

## Running on a SLURM Cluster

Ready-to-use batch scripts for the H100 / H200 nodes live in `scripts/`:

```bash
sbatch scripts/batch_ddp_h100.sh
sbatch scripts/batch_ddp_h200.sh
```

Each script activates the environment, exports `PYTHONPATH=src`, and launches a 2-GPU DDP job
with `torchrun`. Only **rank 0** measures energy, logs metrics and writes checkpoints; the other
ranks stay in lockstep so DDP's gradient all-reduce stays consistent. Adjust the paths, account
and resource directives at the top of the scripts to match your cluster.

---

## Outputs

| Directory | Contents |
|---|---|
| `checkpoints/` | Best model weights (`.pth`) per experiment |
| `outputs/figures/` | Training curves and comparison plots (`.png`) |
| `logs/` | SLURM `stdout`/`stderr` and run logs |

Training reports include **accuracy/loss curves**, plus **CPU + GPU energy** (Joules) and
throughput, enabling accuracy-vs-energy and accuracy-vs-latency trade-off analysis across every
compression technique.

---

## Key Features

- **ViT from scratch** — multi-head attention, encoder blocks, MLP, DropPath, EMA, MixUp, CutMix, focal loss, cosine schedule with warm-up.
- **DALI pipelines** — GPU-side decode/augment with FP16/BF16 output, hardware-decoder tuning, and prefetch queues auto-sized to the GPU.
- **Energy-aware** — Intel RAPL (CPU) and EML/NVML (GPU) instrumentation around every training/eval phase.
- **Full compression toolbox** — quantization (1/2/4/8/16-bit), structured & unstructured pruning, Token Merging, and knowledge distillation.
- **Scales out** — single-GPU, multi-GPU DDP, and SLURM batch jobs from the same codebase.

---

## License & Citation

This project was developed as a **Bachelor's Thesis (Trabajo de Fin de Grado)** at the
**University of La Laguna**, Higher School of Engineering and Technology, for the
Bachelor's Degree in Computer Engineering (2025–2026).

If you reference this work, please cite:

> Sarmiento Rodríguez, Cristóbal Jesús. *High-Performance Computing and Machine Learning:
> Training, Compressing and Profiling Vision Transformers under an Energy Budget.*
> Bachelor's Thesis, University of La Laguna, 2025–2026.

---

<div align="center">

**Universidad de La Laguna**

</div>
