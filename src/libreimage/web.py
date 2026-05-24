from __future__ import annotations

import argparse
import io
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from libreimage.model_store import configure_model_environment
from libreimage.pipelines import (
    DEFAULT_INPAINT_MODEL_ID,
    DEFAULT_KONTEXT_MODEL_ID,
    KontextInpainter,
    KontextInpaintOptions,
    KontextOptions,
    KontextRestorer,
    LocalSharpener,
    SharpenOptions,
)
from libreimage.session_store import SessionStore


DEFAULT_TMP_DIR = Path("tmp") / "libreimage"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(output_dir: Path = DEFAULT_TMP_DIR) -> FastAPI:
    configure_model_environment()
    app = FastAPI(title="LibreImage")
    store = SessionStore(output_dir)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/session")
    async def create_session() -> JSONResponse:
        session_id = store.create_session()
        return JSONResponse({"session_id": session_id, "images": _images_payload(store, session_id)})

    @app.get("/api/session/{session_id}/images")
    async def list_images(session_id: str) -> JSONResponse:
        _require_session(store, session_id)
        return JSONResponse({"images": _images_payload(store, session_id)})

    @app.get("/api/session/{session_id}/images/{image_id}")
    async def image_file(session_id: str, image_id: str) -> FileResponse:
        path = _require_image_path(store, session_id, image_id)
        return FileResponse(path, media_type="image/png", filename=path.name)

    @app.post("/api/session/{session_id}/upload")
    async def upload(session_id: str, image: UploadFile = File(...)) -> JSONResponse:
        _require_session(store, session_id)
        pil = await _read_upload_image(image)
        item = store.add_upload(session_id, pil, image.filename or "upload.png")
        return JSONResponse({"image": _image_payload(item, session_id), "images": _images_payload(store, session_id)})

    @app.delete("/api/session/{session_id}/images/{image_id}")
    async def delete_image(session_id: str, image_id: str) -> JSONResponse:
        _require_session(store, session_id)
        try:
            store.delete_image(session_id, image_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Image not found.") from exc
        return JSONResponse({"images": _images_payload(store, session_id)})

    @app.post("/api/session/{session_id}/restore")
    async def restore(
        session_id: str,
        image_id: str = Form(...),
        prompt: str = Form(KontextOptions.prompt),
        negative_prompt: str = Form(KontextOptions.negative_prompt),
        model: str = Form(DEFAULT_KONTEXT_MODEL_ID),
        steps: int = Form(KontextOptions.steps),
        guidance_scale: float = Form(KontextOptions.guidance_scale),
        strength: float = Form(KontextOptions.strength),
        lora_scale: float = Form(KontextOptions.lora_scale),
        seed: str = Form(""),
    ) -> JSONResponse:
        source = _load_store_image(store, session_id, image_id)
        options = KontextOptions(
            model_id=model.strip() or DEFAULT_KONTEXT_MODEL_ID,
            prompt=prompt.strip() or KontextOptions.prompt,
            negative_prompt=negative_prompt.strip() or KontextOptions.negative_prompt,
            steps=max(1, min(int(steps), 80)),
            guidance_scale=float(guidance_scale),
            strength=max(0.0, min(float(strength), 1.0)),
            lora_scale=max(0.0, min(float(lora_scale), 2.0)),
            seed=_parse_seed(seed),
        )
        result = KontextRestorer(options).run(source)
        item = store.add_image(session_id, result, "restore", "Kontext restore", image_id, asdict(options))
        return JSONResponse({"image": _image_payload(item, session_id), "images": _images_payload(store, session_id)})

    @app.post("/api/session/{session_id}/inpaint")
    async def inpaint(
        session_id: str,
        mask: UploadFile = File(...),
        image_id: str = Form(...),
        prompt: str = Form(KontextInpaintOptions.prompt),
        negative_prompt: str = Form(KontextInpaintOptions.negative_prompt),
        model: str = Form(DEFAULT_INPAINT_MODEL_ID),
        steps: int = Form(KontextInpaintOptions.steps),
        guidance_scale: float = Form(KontextInpaintOptions.guidance_scale),
        strength: float = Form(KontextInpaintOptions.strength),
        seed: str = Form(""),
    ) -> JSONResponse:
        source = _load_store_image(store, session_id, image_id)
        mask_image = await _read_mask(mask, source.size)
        if mask_image.getbbox() is None:
            raise HTTPException(status_code=400, detail="Mask is empty. Paint an area before running inpaint.")
        options = KontextInpaintOptions(
            model_id=model.strip() or DEFAULT_INPAINT_MODEL_ID,
            prompt=prompt.strip() or KontextInpaintOptions.prompt,
            negative_prompt=negative_prompt.strip() or KontextInpaintOptions.negative_prompt,
            steps=max(1, min(int(steps), 80)),
            guidance_scale=float(guidance_scale),
            strength=max(0.0, min(float(strength), 1.0)),
            seed=_parse_seed(seed),
        )
        result = KontextInpainter(options).run(source, mask_image)
        item = store.add_image(session_id, result, "inpaint", "Inpaint repair", image_id, asdict(options))
        return JSONResponse({"image": _image_payload(item, session_id), "images": _images_payload(store, session_id)})

    @app.post("/api/session/{session_id}/sharpen")
    async def sharpen(
        session_id: str,
        image_id: str = Form(...),
        radius: float = Form(SharpenOptions.radius),
        amount: float = Form(SharpenOptions.amount),
        threshold: int = Form(SharpenOptions.threshold),
        contrast: float = Form(SharpenOptions.contrast),
        color: float = Form(SharpenOptions.color),
    ) -> JSONResponse:
        source = _load_store_image(store, session_id, image_id)
        options = SharpenOptions(
            radius=max(0.1, min(float(radius), 5.0)),
            amount=max(0.0, min(float(amount), 4.0)),
            threshold=max(0, min(int(threshold), 32)),
            contrast=max(0.5, min(float(contrast), 1.8)),
            color=max(0.0, min(float(color), 1.8)),
        )
        result = LocalSharpener(options).run(source)
        item = store.add_image(session_id, result, "sharpen", "Sharpen pass", image_id, asdict(options))
        return JSONResponse({"image": _image_payload(item, session_id), "images": _images_payload(store, session_id)})

    return app


def _images_payload(store: SessionStore, session_id: str) -> list[dict[str, object]]:
    return [_image_payload(image, session_id) for image in store.list_images(session_id)]


def _image_payload(image, session_id: str) -> dict[str, object]:
    payload = asdict(image)
    payload["url"] = f"/api/session/{session_id}/images/{image.id}"
    return payload


def _require_session(store: SessionStore, session_id: str) -> None:
    try:
        store.list_images(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


def _require_image_path(store: SessionStore, session_id: str, image_id: str) -> Path:
    try:
        return store.image_path(session_id, image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Image not found.") from exc


def _load_store_image(store: SessionStore, session_id: str, image_id: str) -> Image.Image:
    return Image.open(_require_image_path(store, session_id, image_id)).convert("RGB")


async def _read_upload_image(upload: UploadFile) -> Image.Image:
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(await upload.read()))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


async def _read_mask(upload: UploadFile, size: tuple[int, int]) -> Image.Image:
    try:
        mask = Image.open(io.BytesIO(await upload.read())).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid mask: {exc}") from exc
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask


def _parse_seed(seed: str) -> int | None:
    return int(seed) if seed.strip() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="libreimage", description="Run the LibreImage web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--tmp-dir", default=str(DEFAULT_TMP_DIR), help="Directory for session files.")
    parser.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import uvicorn

    app = create_app(Path(args.tmp_dir))
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
