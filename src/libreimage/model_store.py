from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
HF_HOME = MODELS_DIR / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
SAFETY_CACHE = MODELS_DIR / "safety"


def configure_model_environment() -> None:
    """Keep all downloaded model artifacts inside the repo-local models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    SAFETY_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("DIFFUSERS_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
