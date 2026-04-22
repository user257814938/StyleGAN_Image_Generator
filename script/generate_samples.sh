#!/usr/bin/env bash
set -euo pipefail

NETWORK=""

cd /workspace/stylegan3

python gen_images.py \
  --outdir=/workspace/generated-samples \
  --trunc=1 \
  --seeds=0-31 \
  --network=""
