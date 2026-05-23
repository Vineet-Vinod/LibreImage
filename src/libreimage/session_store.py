from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps


@dataclass(frozen=True)
class LibraryImage:
    id: str
    stage: str
    name: str
    filename: str
    created_at: str
    width: int
    height: int
    parent_id: str | None
    params: dict[str, object]


class SessionStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def create_session(self) -> str:
        session_id = uuid4().hex[:16]
        self._session_dir(session_id).mkdir(parents=True, exist_ok=False)
        self._library_dir(session_id).mkdir(parents=True, exist_ok=True)
        self._write_manifest(session_id, [])
        return session_id

    def ensure_session(self, session_id: str | None) -> str:
        if session_id and self._manifest_path(session_id).exists():
            return session_id
        return self.create_session()

    def list_images(self, session_id: str) -> list[LibraryImage]:
        return [LibraryImage(**item) for item in self._read_manifest(session_id)]

    def image_path(self, session_id: str, image_id: str) -> Path:
        image = self.get_image(session_id, image_id)
        return self._library_dir(session_id) / image.filename

    def get_image(self, session_id: str, image_id: str) -> LibraryImage:
        for image in self.list_images(session_id):
            if image.id == image_id:
                return image
        raise KeyError(image_id)

    def add_upload(self, session_id: str, source: Image.Image, name: str) -> LibraryImage:
        return self.add_image(
            session_id=session_id,
            image=ImageOps.exif_transpose(source).convert("RGB"),
            stage="upload",
            name=name,
            parent_id=None,
            params={},
        )

    def add_image(
        self,
        session_id: str,
        image: Image.Image,
        stage: str,
        name: str,
        parent_id: str | None,
        params: dict[str, object],
    ) -> LibraryImage:
        image_id = uuid4().hex[:12]
        filename = f"{image_id}.png"
        path = self._library_dir(session_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(path, format="PNG")
        item = LibraryImage(
            id=image_id,
            stage=stage,
            name=name,
            filename=filename,
            created_at=datetime.now(timezone.utc).isoformat(),
            width=image.width,
            height=image.height,
            parent_id=parent_id,
            params=params,
        )
        manifest = self._read_manifest(session_id)
        manifest.append(asdict(item))
        self._write_manifest(session_id, manifest)
        return item

    def delete_image(self, session_id: str, image_id: str) -> None:
        manifest = self._read_manifest(session_id)
        kept = [item for item in manifest if item["id"] != image_id]
        if len(kept) == len(manifest):
            raise KeyError(image_id)
        path = self._library_dir(session_id) / next(item["filename"] for item in manifest if item["id"] == image_id)
        if path.exists():
            path.unlink()
        self._write_manifest(session_id, kept)

    def clear_session(self, session_id: str) -> None:
        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)

    def _session_dir(self, session_id: str) -> Path:
        return self.root / "sessions" / session_id

    def _library_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "library"

    def _manifest_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "manifest.json"

    def _read_manifest(self, session_id: str) -> list[dict[str, object]]:
        path = self._manifest_path(session_id)
        if not path.exists():
            raise KeyError(session_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_manifest(self, session_id: str, items: list[dict[str, object]]) -> None:
        path = self._manifest_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, indent=2, sort_keys=True), encoding="utf-8")
