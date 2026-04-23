import io
import os
import sys
import base64
from pathlib import Path

import numpy as np
import PIL.Image
import torch


MODEL_DIR = Path(os.getenv("RUNPOD_MODEL_DIR", "/runpod-volume/stylegan-demo"))
STYLEGAN_DIR = Path(os.getenv("STYLEGAN_REPO_DIR", "/app/stylegan3"))
SNAPSHOT_NAME = os.getenv("STYLEGAN_SNAPSHOT", "network-snapshot-005000.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_stylegan_path() -> None:
    if str(STYLEGAN_DIR) not in sys.path:
        sys.path.insert(0, str(STYLEGAN_DIR))


def _load_generator():
    _ensure_stylegan_path()

    import dnnlib  # type: ignore
    import legacy  # type: ignore

    snapshot_path = MODEL_DIR / SNAPSHOT_NAME
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    with dnnlib.util.open_url(str(snapshot_path)) as f:
        generator = legacy.load_network_pkl(f)["G_ema"].to(DEVICE)

    generator.eval()
    return generator


def _seed_to_image(generator, seed: int, truncation_psi: float = 1.0) -> bytes:
    rng = np.random.RandomState(seed)
    z = torch.from_numpy(rng.randn(1, generator.z_dim)).to(DEVICE)

    label = torch.zeros([1, generator.c_dim], device=DEVICE)
    image = generator(z, label, truncation_psi=truncation_psi, noise_mode="const")
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image_np = image[0].cpu().numpy()

    buffer = io.BytesIO()
    PIL.Image.fromarray(image_np, "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def generate_png_base64(generator, seed: int | None = None, truncation_psi: float = 1.0) -> dict:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big") % 1_000_000_000

    png_bytes = _seed_to_image(generator, seed=seed, truncation_psi=truncation_psi)
    return {
        "seed": seed,
        "image_base64": base64.b64encode(png_bytes).decode("utf-8"),
        "mime_type": "image/png",
    }
