import os

import runpod

from inference import DEVICE, MODEL_DIR, SNAPSHOT_NAME, generate_png_base64, _load_generator


GENERATOR = _load_generator()


def handler(job):
    job_input = job.get("input", {})
    seed = job_input.get("seed")
    truncation_psi = float(job_input.get("truncation_psi", 1.0))

    result = generate_png_base64(
        generator=GENERATOR,
        seed=seed,
        truncation_psi=truncation_psi,
    )

    result.update(
        {
            "device": str(DEVICE),
            "model_dir": str(MODEL_DIR),
            "snapshot": SNAPSHOT_NAME,
            "status": "ok",
        }
    )
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
