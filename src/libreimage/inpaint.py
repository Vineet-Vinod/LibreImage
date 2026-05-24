from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from libreimage.images import clamp_to_multiple_of_eight, load_mask, load_rgb
from libreimage.model_store import HF_HUB_CACHE, resolve_model_path, configure_model_environment
from libreimage.safety import SafetyOptions, ensure_model_checked


DEFAULT_MODEL_ID = "models/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_PROMPT = "natural clean image, realistic texture, seamless restoration"
DEFAULT_NEGATIVE_PROMPT = "text, logo, watermark, label, caption, blurry, distorted"


@dataclass(frozen=True)
class InpaintOptions:
    model_id: str = DEFAULT_MODEL_ID
    revision: str | None = None
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 0.99
    seed: int | None = None
    device: str = "auto"


class LocalInpainter:
    def __init__(self, options: InpaintOptions) -> None:
        self.options = options
        self._pipeline = None
        self._device = None

    @property
    def device(self) -> str | None:
        return self._device

    def run(self, image_path: Path, mask_path: Path, output_path: Path) -> Path:
        image = load_rgb(image_path)
        mask = load_mask(mask_path, image.size)
        result = self.run_images(image, mask)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        return output_path

    def run_images(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        pipeline = self._load_pipeline()
        original_size = image.size
        work_image = clamp_to_multiple_of_eight(image)
        work_mask = clamp_to_multiple_of_eight(mask.convert("L"))

        import torch

        generator = None
        if self.options.seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(self.options.seed)

        with torch.inference_mode():
            result = pipeline(
                prompt=self.options.prompt,
                negative_prompt=self.options.negative_prompt,
                image=work_image,
                mask_image=work_mask,
                num_inference_steps=self.options.steps,
                guidance_scale=self.options.guidance_scale,
                strength=self.options.strength,
                generator=generator,
            ).images[0]

        if result.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
        return Image.composite(result, image, mask.convert("L"))

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        configure_model_environment()
        ensure_model_checked(
            SafetyOptions(
                model_id=resolve_model_path(self.options.model_id),
                revision=self.options.revision,
            )
        )

        import torch
        from diffusers import AutoPipelineForInpainting

        self._device = _resolve_device(self.options.device, torch)
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        load_kwargs = {
            "revision": self.options.revision,
            "torch_dtype": dtype,
            "use_safetensors": True,
            "cache_dir": str(HF_HUB_CACHE),
        }
        if dtype == torch.float16:
            load_kwargs["variant"] = "fp16"
        pipeline = AutoPipelineForInpainting.from_pretrained(resolve_model_path(self.options.model_id), **load_kwargs)
        pipeline = pipeline.to(self._device)
        pipeline.enable_attention_slicing()

        if self._device == "mps":
            torch.mps.empty_cache()

        self._pipeline = pipeline
        return pipeline


def _resolve_device(requested: str, torch_module) -> str:
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"
