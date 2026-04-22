#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/data/ffhq-aligned
python - <<'PY'
from pathlib import Path
from datasets import load_dataset
from PIL import Image, ExifTags
if not hasattr(Image, 'ExifTags'):
    Image.ExifTags = ExifTags
image_dir = Path('/workspace/data/ffhq-aligned')
ds = load_dataset('bitmind/ffhq-256', split='train')
print('Rows:', len(ds), flush=True)
for i, item in enumerate(ds):
    out = image_dir / (str(i).zfill(5) + '.png')
    if not out.exists():
        item['image'].save(out)
    if (i + 1) % 1000 == 0:
        print('saved', i + 1, flush=True)
PY
/workspace/prepare_ffhq.sh
ls -lh /workspace/datasets/ffhq-256x256.zip
