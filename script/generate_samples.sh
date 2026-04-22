#!/usr/bin/env bash
set -euo pipefail

NETWORK="${1:?Usage: $0 /path/to/network-snapshot.pkl}"

cd /workspace/stylegan3

python gen_images.py \
  --outdir=/workspace/generated-samples \
  --trunc=1 \
  --seeds=0-31 \
  --network="$NETWORK"
