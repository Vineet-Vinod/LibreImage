from __future__ import annotations

import os
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
HF_HOME = MODELS_DIR / ".hf"
HF_HUB_CACHE = HF_HOME / "hub"
VENDORED_KONTEXT_MODEL = MODELS_DIR / "FLUX.1-Kontext-dev"
VENDORED_SDXL_INPAINT_MODEL = MODELS_DIR / "stable-diffusion-xl-1.0-inpainting-0.1"
VENDORED_REALESRGAN_X2_MODEL = MODELS_DIR / "Real-ESRGAN" / "RealESRGAN_x2.pth"
KONTEXT_REPO_ID = "black-forest-labs/FLUX.1-Kontext-dev"
SDXL_INPAINT_REPO_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
REALESRGAN_REPO_ID = "ai-forever/Real-ESRGAN"
REALESRGAN_X2_FILENAME = "RealESRGAN_x2.pth"
REALESRGAN_X2_SHA256 = "c830d067d54fc767b9543a8432f36d91bc2de313584e8bbfe4ac26a47339e899"
REALESRGAN_X2_URL = f"https://huggingface.co/{REALESRGAN_REPO_ID}/resolve/main/{REALESRGAN_X2_FILENAME}"


def configure_model_environment() -> None:
    """Keep all downloaded model artifacts inside the repo-local models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("DIFFUSERS_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def ensure_kontext_model() -> Path:
    return _ensure_model(VENDORED_KONTEXT_MODEL, KONTEXT_REPO_ID)


def ensure_inpaint_model() -> Path:
    return _ensure_model(VENDORED_SDXL_INPAINT_MODEL, SDXL_INPAINT_REPO_ID)


def ensure_realesrgan_x2_model() -> Path:
    configure_model_environment()
    if VENDORED_REALESRGAN_X2_MODEL.exists():
        _verify_sha256(VENDORED_REALESRGAN_X2_MODEL, REALESRGAN_X2_SHA256)
        return VENDORED_REALESRGAN_X2_MODEL

    VENDORED_REALESRGAN_X2_MODEL.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = VENDORED_REALESRGAN_X2_MODEL.with_suffix(".pth.tmp")
    _download_file(REALESRGAN_X2_URL, tmp_path)
    _verify_sha256(tmp_path, REALESRGAN_X2_SHA256)
    tmp_path.replace(VENDORED_REALESRGAN_X2_MODEL)
    return VENDORED_REALESRGAN_X2_MODEL


def _ensure_model(local_path: Path, repo_id: str) -> Path:
    configure_model_environment()
    if (local_path / "model_index.json").exists():
        return local_path

    from huggingface_hub import snapshot_download

    local_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_path),
        cache_dir=str(HF_HUB_CACHE),
    )
    return local_path


def _verify_sha256(path: Path, expected: str) -> None:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"Unexpected SHA-256 for {path}: expected {expected}, got {actual}")


def _download_file(url: str, destination: Path) -> None:
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
