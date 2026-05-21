# LibreImage

Local SDXL inpainting for images you own or have rights to edit. LibreImage does not
auto-detect or remove ownership/licensing marks; it works from an explicit mask that
you provide in the CLI or paint in the GUI.

## Setup

```bash
uv sync
```

The default model is `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`.

On Apple Silicon, the pipeline selects MPS automatically when available. CUDA is used
when available on other systems, otherwise CPU is used.

## First model load

Before the host process loads a model for the first time, LibreImage verifies the
model inside a Lima instance. The check rejects unsafe serialized weight formats and
requires safetensors weights.

Start a Lima instance before the first run:

```bash
limactl start default
```

If you are using a model you already trust and want to bypass this gate:

```bash
LIBREIMAGE_SKIP_LIMA_SAFETY=1 uv run libre image.png --mask mask.png
```

or use `--skip-lima-safety`.

## CLI

White mask pixels are replaced. Black mask pixels are preserved.

```bash
uv run libre image.png --mask mask.png
```

The result is written next to the input image as `image_libre.png` by default.

Useful options:

```bash
uv run libre image.png --mask mask.png --steps 40 --seed 123 --prompt "natural clean restoration"
uv run libre image.png --mask mask.png --output restored.png
uv run libre image.png --mask mask.png --device mps
```

## GUI

Open the mask editor:

```bash
uv run libre image.png
```

or:

```bash
uv run libre-gui image.png
```

Paint the area to replace, adjust brush size/settings, then press `Inpaint`. The app
saves the result next to the original image with a `_libre` suffix.
