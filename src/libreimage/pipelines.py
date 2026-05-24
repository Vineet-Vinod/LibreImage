from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from inspect import signature

from PIL import Image, ImageEnhance, ImageFilter

from libreimage.images import clamp_to_multiple_of_eight
from libreimage.model_store import (
    HF_HUB_CACHE,
    VENDORED_KONTEXT_MODEL,
    VENDORED_SDXL_INPAINT_MODEL,
    configure_model_environment,
    resolve_model_path,
)
from libreimage.safety import SafetyOptions, ensure_model_checked


DEFAULT_KONTEXT_MODEL_ID = "models/FLUX.1-Kontext-dev"
DEFAULT_INPAINT_MODEL_ID = "models/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_RESTORE_PROMPT = "Restore the image naturally. Repair damage, remove artifacts, preserve identity, texture, lighting, and composition."
DEFAULT_NEGATIVE_PROMPT = "text, watermark, logo, plastic skin, oversharpening, distorted geometry, extra objects"
DEFAULT_INPAINT_PROMPT = "Natural invisible repair matching the surrounding image."


@dataclass(frozen=True)
class KontextOptions:
    model_id: str = DEFAULT_KONTEXT_MODEL_ID
    prompt: str = DEFAULT_RESTORE_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    steps: int = 24
    guidance_scale: float = 2.5
    strength: float = 0.35
    lora_scale: float = 1.0
    seed: int | None = None
    device: str = "auto"


@dataclass(frozen=True)
class KontextInpaintOptions:
    model_id: str = DEFAULT_INPAINT_MODEL_ID
    prompt: str = DEFAULT_INPAINT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    steps: int = 28
    guidance_scale: float = 3.5
    strength: float = 0.9
    lora_scale: float = 1.0
    seed: int | None = None
    device: str = "auto"


@dataclass(frozen=True)
class SharpenOptions:
    radius: float = 1.4
    amount: float = 1.15
    threshold: int = 3
    contrast: float = 1.02
    color: float = 1.0


class KontextRestorer:
    def __init__(self, options: KontextOptions) -> None:
        self.options = options

    def run(self, image: Image.Image) -> Image.Image:
        configure_model_environment()
        pipe = _load_kontext_pipeline(self.options.model_id, self.options.device)
        work_image = clamp_to_multiple_of_eight(image.convert("RGB"))
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
        }
        if "strength" in signature(pipe.__call__).parameters:
            kwargs["strength"] = self.options.strength
        _add_attention_scale(kwargs, pipe, self.options.lora_scale)
        result = pipe(**kwargs).images[0]
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        return result


class KontextInpainter:
    def __init__(self, options: KontextInpaintOptions) -> None:
        self.options = options

    def run(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        configure_model_environment()
        pipe = _load_inpaint_pipeline(self.options.model_id, self.options.device)
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
        }
        _add_attention_scale(kwargs, pipe, self.options.lora_scale)
        result = pipe(**kwargs).images[0]
        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        return Image.composite(result, image.convert("RGB"), mask.convert("L"))


class LocalSharpener:
    def __init__(self, options: SharpenOptions) -> None:
        self.options = options

    def run(self, image: Image.Image) -> Image.Image:
        result = image.convert("RGB").filter(
            ImageFilter.UnsharpMask(
                radius=self.options.radius,
                percent=max(0, int(self.options.amount * 100)),
                threshold=max(0, int(self.options.threshold)),
            )
        )
        if self.options.contrast != 1.0:
            result = ImageEnhance.Contrast(result).enhance(self.options.contrast)
        if self.options.color != 1.0:
            result = ImageEnhance.Color(result).enhance(self.options.color)
        return result


@lru_cache(maxsize=2)
def _load_kontext_pipeline(model_id: str, device: str):
    resolved_model = resolve_model_path(model_id or VENDORED_KONTEXT_MODEL)
    ensure_model_checked(SafetyOptions(model_id=resolved_model))
    import torch
    from diffusers import FluxKontextPipeline

    resolved_device = _resolve_device(device, torch)
    dtype = _dtype_for_device(resolved_device, torch)
    pipe = FluxKontextPipeline.from_pretrained(
        resolved_model,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=str(HF_HUB_CACHE),
    )
    return _optimize_pipeline(pipe, resolved_device)


@lru_cache(maxsize=2)
def _load_inpaint_pipeline(model_id: str, device: str):
    resolved_model = resolve_model_path(model_id or VENDORED_SDXL_INPAINT_MODEL)
    ensure_model_checked(SafetyOptions(model_id=resolved_model))
    import torch
    from diffusers import AutoPipelineForInpainting

    resolved_device = _resolve_device(device, torch)
    dtype = _dtype_for_device(resolved_device, torch)
    pipe = AutoPipelineForInpainting.from_pretrained(
        resolved_model,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=str(HF_HUB_CACHE),
    )
    return _optimize_pipeline(pipe, resolved_device)


def _optimize_pipeline(pipe, device: str):
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_model_cpu_offload") and device != "mps":
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


def _resolve_device(requested: str, torch_module) -> str:
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _dtype_for_device(device: str, torch_module):
    if device == "cuda":
        return torch_module.float16
    return torch_module.float32
