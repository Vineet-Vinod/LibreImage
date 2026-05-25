from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from urllib.error import HTTPError


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
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
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def ensure_kontext_model() -> Path:
    return _ensure_model(VENDORED_KONTEXT_MODEL, KONTEXT_REPO_ID)


def ensure_sdxl_inpaint_model() -> Path:
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

    local_path.mkdir(parents=True, exist_ok=True)
    _download_repo_files(repo_id, local_path)
    return local_path


def _download_repo_files(repo_id: str, local_path: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    headers = build_hf_headers()
    for filename in HfApi().list_repo_files(repo_id):
        if filename.endswith("/"):
            continue
        destination = local_path / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_name(f"{destination.name}.tmp")
        _download_file(hf_hub_url(repo_id, filename), tmp_path, headers=headers)
        tmp_path.replace(destination)


def _verify_sha256(path: Path, expected: str) -> None:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"Unexpected SHA-256 for {path}: expected {expected}, got {actual}")


def _download_file(url: str, destination: Path, headers: dict[str, str] | None = None) -> None:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request) as response, destination.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
    except HTTPError as exc:
        destination.unlink(missing_ok=True)
        if exc.code in {401, 403}:
            raise RuntimeError(
                "Model download requires Hugging Face access. Run `huggingface-cli login` "
                "or set an HF_TOKEN with access to the requested model."
            ) from exc
        raise
    except Exception:
        destination.unlink(missing_ok=True)
        raise
