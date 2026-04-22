#!/usr/bin/env bash
set -euo pipefail
cd /workspace/stylegan3
python dataset_tool.py --source=/workspace/data/ffhq-aligned --dest=/workspace/datasets/ffhq-256x256.zip --resolution=256x256
