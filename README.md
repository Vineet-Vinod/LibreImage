# LibreImage

LibreImage is a local-network web app for image restoration. The workflow is session based: uploads and generated intermediates are kept
under `tmp/libreimage/`, while only final images you explicitly save are written
to a folder you choose in the browser.

There is no supported CLI workflow. Start the web app and use it from the
browser.

## Setup

```bash
uv sync
```

Model files are stored only inside the repo-local `models/` directory. The app
sets Hugging Face, Diffusers, Transformers, and safety-marker cache environment
variables before loading models so downloads do not go to `~/.cache`.

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

Then open `http://<mac-studio-ip>:7860` from the other device.

You can optionally seed a new session with an image:

```bash
uv run libreimage /path/to/image.png
```

## Workflow

1. Upload an image.
2. Run the Kontext restoration stage. Tune prompt, negative prompt, steps,
   guidance, strength, LoRA scale, seed, and model id. Every successful run is
   saved as a temporary session image.
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

Defaults:

```text
Kontext restore: models/FLUX.1-Kontext-dev
Inpaint:         models/stable-diffusion-xl-1.0-inpainting-0.1
```

The currently vendored models are:

```text
models/FLUX.1-Kontext-dev
models/stable-diffusion-xl-1.0-inpainting-0.1
```

Additional model ids or local model directories can still be entered in the UI.
The backend caches loaded pipelines in process so repeated tuning runs avoid
reloading model weights.

LibreImage still performs the Lima first-load safety check unless you explicitly
enable the per-run "Skip Lima safety" checkbox for a trusted model.

## Apple Silicon Performance

The app automatically selects MPS when available, enables PyTorch MPS fallback,
keeps pipeline instances warm in memory, and avoids CPU offload on MPS. For best
throughput, keep the server running while tuning so the loaded model cache stays
hot.
