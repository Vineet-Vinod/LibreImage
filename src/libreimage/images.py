from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def load_rgb(path: str | Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def load_mask(path: str | Path, size: tuple[int, int]) -> Image.Image:
    mask = ImageOps.exif_transpose(Image.open(path)).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask


def clamp_to_multiple_of_eight(image: Image.Image) -> Image.Image:
    width, height = image.size
    adjusted = (max(8, width - width % 8), max(8, height - height % 8))
    if adjusted == image.size:
        return image
    return image.resize(adjusted, Image.Resampling.LANCZOS)
