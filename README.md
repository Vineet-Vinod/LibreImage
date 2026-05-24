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
   guidance, strength, and seed. Every successful run is saved as a temporary
   session image.
3. Move to Inpaint. Select any saved image from the bottom filmstrip, paint a
   mask with the brush or eraser, tune the inpaint params, and save more
   intermediates.
4. Move to Sharpen. Select any intermediate, tune the local sharpening controls,
   and save sharpened variants.
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
```

If either directory is missing, the app downloads that model from Hugging Face
into `models/` before loading it. The backend caches loaded pipelines in process
so repeated tuning runs avoid reloading model weights.

Note: The performance has been tuned for Apple Silicon. It may not be optimal on
GPUs. Happy to merge PRs that improve GPU performance.
