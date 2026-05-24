from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from libreimage.model_store import SAFETY_CACHE, configure_model_environment


UNSAFE_MODEL_SUFFIXES = (".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle")


@dataclass(frozen=True)
class SafetyOptions:
    model_id: str
    revision: str | None = None
    cache_dir: Path = SAFETY_CACHE


class ModelSafetyError(RuntimeError):
    """Raised when a local model directory contains unsafe weight formats."""


def ensure_model_checked(options: SafetyOptions) -> Path | None:
    configure_model_environment()
    local_path = Path(options.model_id).expanduser()
    if not local_path.exists():
        return None

    marker_path = _marker_path(options)
    if marker_path.exists():
        return marker_path

    payload = _verify_local_model(local_path)
    _write_marker(marker_path, options, payload)
    return marker_path


def _marker_path(options: SafetyOptions) -> Path:
    key = f"{options.model_id}@{options.revision or 'main'}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return options.cache_dir / f"{digest}.json"


def _verify_local_model(model_path: Path) -> dict[str, object]:
    unsafe = [
        str(path.relative_to(model_path))
        for suffix in UNSAFE_MODEL_SUFFIXES
        for path in model_path.rglob(f"*{suffix}")
    ]
    if unsafe:
        raise ModelSafetyError("Refusing local model with unsafe serialized weights: " + ", ".join(unsafe))
    safetensors = [str(path.relative_to(model_path)) for path in model_path.rglob("*.safetensors")]
    if not safetensors:
        raise ModelSafetyError(f"Local model does not contain safetensors weights: {model_path}")
    return {"local_path": str(model_path), "safe_tensors": safetensors, "file_count": len(list(model_path.rglob("*")))}


def _write_marker(marker_path: Path, options: SafetyOptions, payload: dict[str, object]) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker = {
        "model_id": options.model_id,
        "revision": options.revision,
        "payload": payload,
    }
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
