from __future__ import annotations

import argparse
import base64
import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from libreimage.inpaint import DEFAULT_MODEL_ID, InpaintOptions, LocalInpainter


def create_app() -> FastAPI:
    app = FastAPI(title="LibreImage")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    @app.post("/api/inpaint")
    async def inpaint(
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        prompt: str = Form(InpaintOptions.prompt),
        negative_prompt: str = Form(InpaintOptions.negative_prompt),
        steps: int = Form(InpaintOptions.steps),
        guidance_scale: float = Form(InpaintOptions.guidance_scale),
        strength: float = Form(InpaintOptions.strength),
        seed: str = Form(""),
        model: str = Form(DEFAULT_MODEL_ID),
    ) -> JSONResponse:
        try:
            image_pil = Image.open(io.BytesIO(await image.read())).convert("RGB")
            mask_pil = Image.open(io.BytesIO(await mask.read())).convert("L")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image or mask: {exc}") from exc

        if mask_pil.size != image_pil.size:
            mask_pil = mask_pil.resize(image_pil.size, Image.Resampling.NEAREST)
        if mask_pil.getbbox() is None:
            raise HTTPException(status_code=400, detail="Mask is empty. Paint an area before running inpaint.")

        parsed_seed = int(seed) if seed.strip() else None
        options = InpaintOptions(
            model_id=model.strip() or DEFAULT_MODEL_ID,
            prompt=prompt.strip() or InpaintOptions.prompt,
            negative_prompt=negative_prompt.strip() or InpaintOptions.negative_prompt,
            steps=max(1, min(int(steps), 100)),
            guidance_scale=float(guidance_scale),
            strength=max(0.0, min(float(strength), 1.0)),
            seed=parsed_seed,
        )

        try:
            result = LocalInpainter(options).run_images(image_pil, mask_pil)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return JSONResponse({"image": f"data:image/png;base64,{encoded}"})

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="libre-web", description="Run the LibreImage browser UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import uvicorn

    uvicorn.run(
        "libreimage.web:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LibreImage</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #14161a;
      --panel: #20242a;
      --panel-2: #252b33;
      --text: #edf1f7;
      --muted: #a9b2c3;
      --line: #3a4250;
      --accent: #4fb7ff;
      --danger: #ff6d6d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      min-height: 100vh;
    }
    .workspace {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-width: 0;
    }
    .topbar, .sidebar {
      background: var(--panel);
      border-color: var(--line);
    }
    .topbar {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      min-width: 0;
    }
    .brand {
      font-weight: 700;
      margin-right: 8px;
      white-space: nowrap;
    }
    .toolgroup {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }
    .canvas-wrap {
      display: grid;
      place-items: center;
      overflow: hidden;
      background: #0f1115;
      min-height: 0;
      position: relative;
    }
    #stage {
      max-width: 100%;
      max-height: 100%;
      touch-action: none;
      background: #111;
      box-shadow: 0 0 0 1px #000;
    }
    .sidebar {
      border-left: 1px solid var(--line);
      padding: 14px;
      overflow: auto;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin: 14px 0 6px;
    }
    input[type="text"], input[type="number"], textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      color: var(--text);
      padding: 9px 10px;
      font: inherit;
    }
    textarea { min-height: 74px; resize: vertical; }
    input[type="file"] { max-width: 100%; color: var(--muted); }
    input[type="range"] { width: 160px; }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      color: var(--text);
      padding: 8px 11px;
      font: inherit;
      cursor: pointer;
    }
    button.active { border-color: var(--accent); color: #dff3ff; }
    button.primary {
      width: 100%;
      margin-top: 16px;
      background: var(--accent);
      border-color: var(--accent);
      color: #05131e;
      font-weight: 700;
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .row > * { flex: 1; }
    .status {
      margin-top: 12px;
      min-height: 20px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .output {
      margin-top: 14px;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      display: none;
    }
    .empty {
      color: var(--muted);
      position: absolute;
      text-align: center;
      padding: 24px;
      pointer-events: none;
    }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; grid-template-rows: minmax(360px, 1fr) auto; }
      .sidebar { border-left: 0; border-top: 1px solid var(--line); max-height: 44vh; }
      .topbar { flex-wrap: wrap; }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="workspace">
      <div class="topbar">
        <div class="brand">LibreImage</div>
        <input id="imageInput" type="file" accept="image/*">
        <div class="toolgroup">
          <button id="brushBtn" class="active" type="button">Brush</button>
          <button id="eraseBtn" type="button">Erase</button>
          <button id="clearBtn" type="button">Clear</button>
        </div>
        <div class="toolgroup">
          <span>Size</span>
          <input id="brushSize" type="range" min="4" max="180" value="48">
          <span id="brushSizeValue">48</span>
        </div>
      </div>
      <div class="canvas-wrap">
        <canvas id="stage"></canvas>
        <div id="empty" class="empty">Open an image to paint a mask.</div>
      </div>
    </section>
    <aside class="sidebar">
      <label for="prompt">Prompt</label>
      <textarea id="prompt">natural clean image, realistic texture, seamless restoration</textarea>
      <label for="negativePrompt">Negative Prompt</label>
      <textarea id="negativePrompt">text, logo, watermark, label, caption, blurry, distorted</textarea>
      <label for="model">Model</label>
      <input id="model" type="text" value="diffusers/stable-diffusion-xl-1.0-inpainting-0.1">
      <div class="row">
        <div>
          <label for="steps">Steps</label>
          <input id="steps" type="number" min="1" max="100" value="30">
        </div>
        <div>
          <label for="seed">Seed</label>
          <input id="seed" type="text" placeholder="random">
        </div>
      </div>
      <div class="row">
        <div>
          <label for="guidance">Guidance</label>
          <input id="guidance" type="number" min="1" max="20" step="0.5" value="7.5">
        </div>
        <div>
          <label for="strength">Strength</label>
          <input id="strength" type="number" min="0.1" max="1" step="0.05" value="0.99">
        </div>
      </div>
      <button id="runBtn" class="primary" type="button">Inpaint</button>
      <div id="status" class="status"></div>
      <a id="download" class="status" download="libreimage-result.png" href="#" style="display:none">Download result</a>
      <img id="output" class="output" alt="Inpaint result">
    </aside>
  </main>
  <script>
    const imageInput = document.getElementById("imageInput");
    const canvas = document.getElementById("stage");
    const ctx = canvas.getContext("2d");
    const empty = document.getElementById("empty");
    const brushBtn = document.getElementById("brushBtn");
    const eraseBtn = document.getElementById("eraseBtn");
    const clearBtn = document.getElementById("clearBtn");
    const brushSize = document.getElementById("brushSize");
    const brushSizeValue = document.getElementById("brushSizeValue");
    const runBtn = document.getElementById("runBtn");
    const statusEl = document.getElementById("status");
    const download = document.getElementById("download");
    const output = document.getElementById("output");

    let sourceFile = null;
    let sourceImage = null;
    let maskCanvas = document.createElement("canvas");
    let maskCtx = maskCanvas.getContext("2d");
    let painting = false;
    let mode = "paint";
    let scale = 1;
    let offsetX = 0;
    let offsetY = 0;
    let last = null;

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.style.color = isError ? "var(--danger)" : "var(--muted)";
    }

    function setMode(next) {
      mode = next;
      brushBtn.classList.toggle("active", mode === "paint");
      eraseBtn.classList.toggle("active", mode === "erase");
    }

    function resizeStage() {
      const wrap = canvas.parentElement.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(wrap.width));
      canvas.height = Math.max(1, Math.floor(wrap.height));
      redraw();
    }

    function redraw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!sourceImage) {
        empty.style.display = "block";
        return;
      }
      empty.style.display = "none";
      scale = Math.min(canvas.width / sourceImage.width, canvas.height / sourceImage.height, 1);
      const drawW = Math.round(sourceImage.width * scale);
      const drawH = Math.round(sourceImage.height * scale);
      offsetX = Math.floor((canvas.width - drawW) / 2);
      offsetY = Math.floor((canvas.height - drawH) / 2);
      ctx.drawImage(sourceImage, offsetX, offsetY, drawW, drawH);
      ctx.save();
      ctx.globalAlpha = 0.42;
      ctx.fillStyle = "#ff4848";
      ctx.drawImage(maskCanvas, offsetX, offsetY, drawW, drawH);
      ctx.globalCompositeOperation = "source-atop";
      ctx.fillRect(offsetX, offsetY, drawW, drawH);
      ctx.restore();
    }

    function canvasToImage(event) {
      if (!sourceImage) return null;
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left - offsetX) / scale;
      const y = (event.clientY - rect.top - offsetY) / scale;
      if (x < 0 || y < 0 || x >= sourceImage.width || y >= sourceImage.height) return null;
      return { x, y };
    }

    function drawStroke(a, b) {
      if (!a || !b) return;
      const width = Number(brushSize.value);
      maskCtx.save();
      maskCtx.lineCap = "round";
      maskCtx.lineJoin = "round";
      maskCtx.lineWidth = width;
      maskCtx.strokeStyle = mode === "paint" ? "#fff" : "#000";
      maskCtx.fillStyle = maskCtx.strokeStyle;
      maskCtx.beginPath();
      maskCtx.moveTo(a.x, a.y);
      maskCtx.lineTo(b.x, b.y);
      maskCtx.stroke();
      maskCtx.beginPath();
      maskCtx.arc(b.x, b.y, width / 2, 0, Math.PI * 2);
      maskCtx.fill();
      maskCtx.restore();
      redraw();
    }

    imageInput.addEventListener("change", () => {
      const file = imageInput.files[0];
      if (!file) return;
      sourceFile = file;
      const image = new Image();
      image.onload = () => {
        sourceImage = image;
        maskCanvas.width = image.width;
        maskCanvas.height = image.height;
        maskCtx.fillStyle = "#000";
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
        output.style.display = "none";
        setStatus(`${file.name} loaded. Paint the areas to replace.`);
        redraw();
      };
      image.src = URL.createObjectURL(file);
    });

    canvas.addEventListener("pointerdown", (event) => {
      const point = canvasToImage(event);
      if (!point) return;
      painting = true;
      last = point;
      canvas.setPointerCapture(event.pointerId);
      drawStroke(point, point);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!painting) return;
      const point = canvasToImage(event);
      if (!point) return;
      drawStroke(last, point);
      last = point;
    });
    canvas.addEventListener("pointerup", () => { painting = false; last = null; });
    canvas.addEventListener("pointercancel", () => { painting = false; last = null; });

    brushBtn.addEventListener("click", () => setMode("paint"));
    eraseBtn.addEventListener("click", () => setMode("erase"));
    clearBtn.addEventListener("click", () => {
      if (!sourceImage) return;
      maskCtx.fillStyle = "#000";
      maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
      redraw();
    });
    brushSize.addEventListener("input", () => { brushSizeValue.textContent = brushSize.value; });
    window.addEventListener("resize", resizeStage);

    runBtn.addEventListener("click", async () => {
      if (!sourceFile || !sourceImage) {
        setStatus("Open an image first.", true);
        return;
      }
      const maskBlob = await new Promise(resolve => maskCanvas.toBlob(resolve, "image/png"));
      const form = new FormData();
      form.append("image", sourceFile, sourceFile.name);
      form.append("mask", maskBlob, "mask.png");
      form.append("prompt", document.getElementById("prompt").value);
      form.append("negative_prompt", document.getElementById("negativePrompt").value);
      form.append("model", document.getElementById("model").value);
      form.append("steps", document.getElementById("steps").value);
      form.append("guidance_scale", document.getElementById("guidance").value);
      form.append("strength", document.getElementById("strength").value);
      form.append("seed", document.getElementById("seed").value);

      runBtn.disabled = true;
      setStatus("Running inpaint. Model loading may take a moment.");
      try {
        const response = await fetch("/api/inpaint", { method: "POST", body: form });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Inpaint failed.");
        output.src = payload.image;
        download.href = payload.image;
        download.style.display = "block";
        output.style.display = "block";
        setStatus("Done.");
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        runBtn.disabled = false;
      }
    });

    resizeStage();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
