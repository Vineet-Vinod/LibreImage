from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from inspect import signature

from PIL import Image, ImageOps

from libreimage.images import clamp_to_multiple_of_eight
from libreimage.model_store import (
    configure_model_environment,
    ensure_sdxl_inpaint_model,
    ensure_kontext_model,
    ensure_realesrgan_x2_model,
)
from libreimage.realesrgan import RealESRGAN2x, RealESRGANConfig


DEFAULT_RESTORE_PROMPT = "Restore the image naturally. Repair damage, remove artifacts, preserve identity, texture, lighting, and composition."
DEFAULT_NEGATIVE_PROMPT = "text, watermark, logo, plastic skin, oversharpening, distorted geometry, extra objects"
DEFAULT_INPAINT_PROMPT = "Natural invisible repair matching the surrounding image."
KONTEXT_MAX_AREA = 1024**2
KONTEXT_RESOLUTION_MULTIPLE = 16


@dataclass(frozen=True)
class KontextOptions:
    prompt: str = DEFAULT_RESTORE_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    steps: int = 24
    guidance_scale: float = 2.5
    strength: float = 0.35
    lora_scale: float = 1.0
    seed: int | None = None
    device: str = "auto"


@dataclass(frozen=True)
class SDXLInpaintOptions:
    prompt: str = DEFAULT_INPAINT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    steps: int = 28
    guidance_scale: float = 3.5
    strength: float = 0.9
    seed: int | None = None
    device: str = "auto"


@dataclass(frozen=True)
class SharpenOptions:
    tile_size: int = 320
    tile_pad: int = 24
    batch_size: int = 8
    device: str = "auto"


class KontextRestorer:
    def __init__(self, options: KontextOptions) -> None:
        self.options = options

    def run(self, image: Image.Image) -> Image.Image:
        configure_model_environment()
        pipe = _load_kontext_pipeline(self.options.device)
        work_size = _kontext_work_size(image.size)
        work_image = ImageOps.fit(image.convert("RGB"), work_size, Image.Resampling.LANCZOS)
        generator = _generator(self.options.seed, pipe.device.type)
        kwargs = {
            "image": work_image,
            "prompt": self.options.prompt,
            "negative_prompt": self.options.negative_prompt,
            "num_inference_steps": self.options.steps,
            "guidance_scale": self.options.guidance_scale,
            "generator": generator,
            "height": work_image.height,
            "width": work_image.width,
            "max_area": work_image.width * work_image.height,
            "_auto_resize": False,
        }
        if "strength" in signature(pipe.__call__).parameters:
            kwargs["strength"] = self.options.strength
        _add_attention_scale(kwargs, pipe, self.options.lora_scale)
        result = pipe(**kwargs).images[0]
        if result.size != image.size:
            result = ImageOps.fit(result, image.size, Image.Resampling.LANCZOS)
        return result


class SDXLInpainter:
    def __init__(self, options: SDXLInpaintOptions) -> None:
        self.options = options

    def run(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        configure_model_environment()
        pipe = _load_sdxl_inpaint_pipeline(self.options.device)
        original_size = image.size
        work_image = clamp_to_multiple_of_eight(image.convert("RGB"))
        work_mask = clamp_to_multiple_of_eight(mask.convert("L"))
        generator = _generator(self.options.seed, pipe.device.type)
        kwargs = {
            "image": work_image,
            "mask_image": work_mask,
            "prompt": self.options.prompt,
            "negative_prompt": self.options.negative_prompt,
            "num_inference_steps": self.options.steps,
            "guidance_scale": self.options.guidance_scale,
            "strength": self.options.strength,
            "generator": generator,
            "height": work_image.height,
            "width": work_image.width,
        }
        result = pipe(**kwargs).images[0]
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        return Image.composite(result, image.convert("RGB"), mask.convert("L"))


class RealESRGANSharpener:
    def __init__(self, options: SharpenOptions) -> None:
        self.options = options

    def run(self, image: Image.Image) -> Image.Image:
        configure_model_environment()
        model = _load_realesrgan_x2(self.options.device)
        config = RealESRGANConfig(
            tile_size=self.options.tile_size,
            tile_pad=self.options.tile_pad,
            batch_size=self.options.batch_size,
        )
        return model.run(image, config)


@lru_cache(maxsize=1)
def _load_kontext_pipeline(device: str):
    model_path = ensure_kontext_model()
    import torch
    from diffusers import FluxKontextPipeline

    resolved_device = _resolve_device(device, torch)
    dtype = _dtype_for_device(resolved_device, torch)
    pipe = FluxKontextPipeline.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    return _optimize_pipeline(pipe, resolved_device)


@lru_cache(maxsize=1)
def _load_sdxl_inpaint_pipeline(device: str):
    model_path = ensure_sdxl_inpaint_model()
    import torch
    from diffusers import AutoPipelineForInpainting

    resolved_device = _resolve_device(device, torch)
    dtype = _dtype_for_device(resolved_device, torch)
    pipe = AutoPipelineForInpainting.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    return _optimize_pipeline(pipe, resolved_device)


@lru_cache(maxsize=1)
def _load_realesrgan_x2(device: str) -> RealESRGAN2x:
    model_path = ensure_realesrgan_x2_model()
    import torch

    resolved_device = _resolve_device(device, torch)
    dtype = _dtype_for_device(resolved_device, torch)
    return RealESRGAN2x(model_path, resolved_device, dtype)


def _optimize_pipeline(pipe, device: str):
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    elif hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    try:
        pipe.unet.to(memory_format=__import__("torch").channels_last)
    except AttributeError:
        pass
    if device == "mps":
        import torch

        torch.mps.empty_cache()
    return pipe


def _generator(seed: int | None, device: str):
    if seed is None:
        return None
    import torch

    return torch.Generator(device=device).manual_seed(seed)


def _add_attention_scale(kwargs: dict[str, object], pipe, scale: float) -> None:
    if scale == 1.0:
        return
    parameters = signature(pipe.__call__).parameters
    if "joint_attention_kwargs" in parameters:
        kwargs["joint_attention_kwargs"] = {"scale": scale}
    elif "cross_attention_kwargs" in parameters:
        kwargs["cross_attention_kwargs"] = {"scale": scale}


def _kontext_work_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    aspect_ratio = width / height
    work_width = round((KONTEXT_MAX_AREA * aspect_ratio) ** 0.5)
    work_height = round((KONTEXT_MAX_AREA / aspect_ratio) ** 0.5)
    work_width = max(
        KONTEXT_RESOLUTION_MULTIPLE,
        work_width // KONTEXT_RESOLUTION_MULTIPLE * KONTEXT_RESOLUTION_MULTIPLE,
    )
    work_height = max(
        KONTEXT_RESOLUTION_MULTIPLE,
        work_height // KONTEXT_RESOLUTION_MULTIPLE * KONTEXT_RESOLUTION_MULTIPLE,
    )
    return work_width, work_height


def _resolve_device(requested: str, torch_module) -> str:
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _dtype_for_device(device: str, torch_module):
    if device in {"cuda", "mps"}:
        return torch_module.float32
    return torch_module.float32
