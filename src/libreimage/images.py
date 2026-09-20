from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageCms, ImageOps


_SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
SRGB_ICC_PROFILE = _SRGB_PROFILE.tobytes()


def convert_to_srgb(image: Image.Image) -> Image.Image:
    transposed = ImageOps.exif_transpose(image)
    icc_profile = transposed.info.get("icc_profile")
    if isinstance(icc_profile, bytes):
        try:
            source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
            converted = ImageCms.profileToProfile(
                transposed,
                source_profile,
                _SRGB_PROFILE,
                outputMode="RGB",
            )
            converted.info["icc_profile"] = SRGB_ICC_PROFILE
            return converted
        except (ImageCms.PyCMSError, OSError, ValueError):
            pass

    converted = transposed.convert("RGB")
    converted.info["icc_profile"] = SRGB_ICC_PROFILE
    return converted


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return convert_to_srgb(image)


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
