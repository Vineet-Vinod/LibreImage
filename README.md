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

## Web UI

Start the browser mask editor:

```bash
uv run libre image.png
```

or:

```bash
uv run libre-web image.png
```

For remote development, bind to all interfaces and forward the port from your editor
or SSH session:

```bash
uv run libre-web --host 0.0.0.0 --port 7860
```

Open the forwarded URL in your browser, upload an image, paint the mask, then press
`Inpaint`. The result appears in the browser with a download link.

For multiple passes, press `Use Result` after a run completes. The generated image
becomes the new source image, the mask is cleared, and the next pass is saved as a
separate run.

Each web inpaint run is also saved locally under the gitignored directory
`tmp/libreimage/runs/<run_id>/`:

```text
source.png
mask.png
result.png
meta.json
```

Use `--tmp-dir` to store those run files somewhere else:

```bash
uv run libre-web image.png --tmp-dir /path/to/runs
```
