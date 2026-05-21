from __future__ import annotations

from pathlib import Path


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def require_image_path(path: str | Path) -> Path:
    image_path = Path(path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")
    return image_path


def default_output_path(image_path: Path, suffix: str = "_libre") -> Path:
    return image_path.with_name(f"{image_path.stem}{suffix}{image_path.suffix}")
