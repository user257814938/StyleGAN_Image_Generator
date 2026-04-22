#!/usr/bin/env bash
set -euo pipefail
cd /workspace/stylegan3
python train.py --outdir=/workspace/training-runs --cfg=stylegan2 --data=/workspace/datasets/ffhq-256x256.zip --gpus=1 --batch=64 --batch-gpu=64 --gamma=0.2048 --mirror=1 --aug=noaug --metrics=none --kimg=5000 --snap=50 --tick=4 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --mbstd-group=4 --dry-run
