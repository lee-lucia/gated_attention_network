```
# Gated Attention Network
A Lightweight Gated Attention Network

## 📖 Introduction
This repository contains the official PyTorch implementation of the dual-branch enhancement network for underwater polarization image enhancement.
The model supports polarization data loading, multi-modal feature fusion, and objective image quality evaluation (PSNR, SSIM).

## 💻 Environment
- OS: Ubuntu 20.04 / Linux
- Python: 3.8.x (Tested on 3.8.10)
- PyTorch: 1.11.0 (Requires CUDA 11.3)
- Torchvision: 0.12.0

> ⚠️ Please strictly follow the version configuration to avoid compatibility errors.

## 📦 Dependencies
| Library | Version | Function |
| ---- | ---- | ---- |
| opencv-python-headless | >=4.13.0 | Basic image reading and matrix transformation |
| scikit-image | ==0.18.3 | Calculate objective evaluation metrics: PSNR, SSIM |
| Pillow | ==9.1.1 | Image preprocessing, cooperate with Torchvision |
| numpy | ==1.22.4 | Numerical calculation and array operation |
| scipy | ==1.10.1 | Scientific calculation and feature processing |
| pytorch-msssim | ==1.0.0 | SSIM loss function based on PyTorch tensor |
| tensorboard | ==2.9.1 | Training loss curve visualization |
| torchsummary / thop | Latest | Count model Params and FLOPs |
| tqdm | ==4.61.2 | Progress bar for training and validation |
| einops | ==0.8.1 | Efficient tensor dimension rearrangement |

### Install
1. Create conda environment
```bash
conda create -n gated_attention python=3.8.10
conda activate gated_attention
```

2. Install PyTorch (CUDA 11.3)

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

3. Install other dependencies

```
pip install opencv-python-headless>=4.13.0
pip install scikit-image==0.18.3 Pillow==9.1.1
pip install numpy==1.22.4 scipy==1.10.1
pip install pytorch-msssim==1.0.0 tensorboard==2.9.1
pip install torchsummary thop
pip install tqdm==4.61.2 einops==0.8.1
```

Or install with `requirements.txt`

```
pip install -r requirements.txt
```

## 📁 Project Structure

```
dual_branch_enhance/
├── config/                 # Configuration files
├── src/                    # Core source code
│   ├── __init__.py
│   ├── data.py             # Dataset loading
│   ├── model.py            # Enhancement network
│   ├── losses.py           # Custom loss functions
│   └── metrics.py          # Evaluation metrics
├── benchmark.py            # Params & FLOPs benchmark script
├── train.py                # Training entry script
├── test.py                 # Test & evaluation entry script
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Quick Start

### 1. Dataset Preparation

The input image and ground-truth reference are matched by filename stem (e.g. `001.jpg` matches `001.png`).
Each image pair must have identical original resolution.

Three supported folder structures:

```
# LSUI format
dataset/
├── input/
└── GT/

# UIEB format
dataset/
├── raw/
└── reference/

# Pre-split train/validation format
dataset/
├── train/
│   ├── input/
│   └── target/
└── val/
    ├── input/
    └── target/
```

### 2. Train

```
python train.py --data "../LSUI" --save-dir "runs/lgan" --epochs 100 --batch-size 32 --workers 0
```

### 3. Test & Evaluate

```bash
python test.py --input "../LSUI/input" --target "../LSUI/GT" --checkpoint "runs/lgan/best.pth" --split "runs/lgan/split.json" --output "runs/lgan/test_output"
```

### 4. Benchmark (Calculate Params and FLOPs)

```bash
python metrics.py --pred "runs/lgan/enhanced_best" --target "../LSUI/GT" --split "runs/lgan/split.json" --resize 256 256
```

## ⚠️ Notes

1. Do not upload cache folders (`.ipynb_checkpoints`, `__pycache__`, `.sys`) to git.
2. If you encounter version conflict, rebuild the conda environment.
3. Check GPU memory before training, adjust batch size according to your hardware.