# LibreImage

LibreImage is a local-network web app for image restoration. The workflow is session 
based: uploads and generated intermediates are kept under `tmp/libreimage/`,
while only final images you explicitly save are written to a folder you choose
in the browser.

## Setup

```bash
uv sync
```

Model files are stored only inside the repo-local `models/` directory. On first
use, the app checks for the bundled model directories there and downloads any
missing model into `models/`.

`models/` and `tmp/` are gitignored.

## System Requirements

LibreImage is tuned for Apple Silicon with PyTorch MPS. The bundled models are
large: Flux Kontext is the limiting stage and loaded in float16 uses about 33 GB
of MPS allocation in smoke tests on an M3 Ultra. Attention and VAE slicing are
enabled by default; this is slower than the unsliced path in small benchmarks but
produced better text-removal results during restoration testing.

Recommended:

- Apple Silicon Mac with MPS support.
- 64 GB unified memory minimum for practical Flux Kontext use.
- 128 GB or more unified memory for comfortable tuning and keeping both
  pipelines warm.
- 100 GB or more free disk space. The bundled local model directories are about
  51 GB combined, before temporary outputs and cache growth.

CPU fallback is available through PyTorch but is expected to be very slow. CUDA
may work through PyTorch on suitable hardware, but this project is not tuned or
tested for CUDA.

## Run

```bash
uv run libreimage
```

Open:

```text
http://127.0.0.1:7860
```

To use the app from another device on the local network:

```bash
uv run libreimage --host 0.0.0.0 --port 7860
```

Then open `http://<server-ip>:7860` from the other device.

## Workflow

1. Upload an image.
2. Run the Kontext restoration stage. Tune prompt, negative prompt, steps,
   guidance, strength, LoRA scale, and seed. Every successful run is saved as a
   temporary session image.
3. Move to Inpaint. Select any saved image from the bottom filmstrip, paint a
   mask with the brush or eraser, tune the inpaint params, and save more
   intermediates.
4. Move to Sharpen. Select any intermediate and run the Real-ESRGAN 2x model to
   upscale and sharpen it. The default tile settings are tuned on
   `tmp/test.png` for Apple Silicon MPS: tile `320`, overlap `24`, batch `8`.
5. Use Final to review every image in a slideshow. Arrow keys navigate between
   images. Save the current image or all images to a browser-selected folder.
   You can restart at the Kontext stage from the currently selected final image
   without clearing the rest of the session library.

Intermediate images can be deleted from the Kontext, Inpaint, Sharpen, and Final
views.

## Models

LibreImage uses these fixed local model directories:

```text
Kontext restore: models/FLUX.1-Kontext-dev
Inpaint:         models/stable-diffusion-xl-1.0-inpainting-0.1
Sharpen:         models/Real-ESRGAN/RealESRGAN_x2.pth
```

If a model is missing, the app downloads it from Hugging Face into `models/`
before loading it. The sharpen stage always uses the ai-forever Real-ESRGAN 2x
checkpoint and verifies its SHA-256 before loading it with PyTorch's restricted
`weights_only=True` checkpoint loader. The backend caches loaded pipelines in
process so repeated tuning runs avoid reloading model weights.

Real-ESRGAN sharpen tuning:

```text
Tile:    larger values reduce tile count and are faster until memory pressure rises.
Overlap: larger values reduce seams but increase work per tile.
Batch:   larger values process more tiles together when memory allows.
```
