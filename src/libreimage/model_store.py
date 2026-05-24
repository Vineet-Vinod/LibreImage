from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
HF_HOME = MODELS_DIR / ".hf"
HF_HUB_CACHE = HF_HOME / "hub"
VENDORED_KONTEXT_MODEL = MODELS_DIR / "FLUX.1-Kontext-dev"
VENDORED_SDXL_INPAINT_MODEL = MODELS_DIR / "stable-diffusion-xl-1.0-inpainting-0.1"
KONTEXT_REPO_ID = "black-forest-labs/FLUX.1-Kontext-dev"
SDXL_INPAINT_REPO_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


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
