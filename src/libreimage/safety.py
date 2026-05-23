from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

from libreimage.model_store import SAFETY_CACHE, configure_model_environment


DEFAULT_LIMA_INSTANCE = "default"
UNSAFE_MODEL_SUFFIXES = (".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle")


@dataclass(frozen=True)
class SafetyOptions:
    model_id: str
    revision: str | None = None
    lima_instance: str = DEFAULT_LIMA_INSTANCE
    cache_dir: Path = SAFETY_CACHE
    skip_lima: bool = False


class ModelSafetyError(RuntimeError):
    """Raised when a model has not passed first-load isolation."""


def ensure_model_checked(options: SafetyOptions) -> Path:
    configure_model_environment()
    marker_path = _marker_path(options)
    if marker_path.exists():
        return marker_path

    if options.skip_lima or os.environ.get("LIBREIMAGE_SKIP_LIMA_SAFETY") == "1":
        _write_marker(marker_path, options, {"skipped": True})
        return marker_path

    script = _verification_script(options.model_id, options.revision)
    command = [
        "limactl",
        "shell",
        options.lima_instance,
        "bash",
        "-lc",
        _lima_python_command(),
    ]
    try:
        result = subprocess.run(
            command,
            input=script,
            text=True,
            capture_output=True,
            check=False,
            timeout=2 * 60 * 60,
        )
    except FileNotFoundError as exc:
        raise ModelSafetyError(
            "First model load must be verified in Lima, but `limactl` was not found. "
            "Install Lima and start an instance, or set LIBREIMAGE_SKIP_LIMA_SAFETY=1 "
            "only for models you already trust."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ModelSafetyError("Timed out while verifying the model inside Lima.") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "no output"
        raise ModelSafetyError(
            "Lima model verification failed for "
            f"{options.model_id!r} on instance {options.lima_instance!r}.\n{details}"
        )

    payload = _parse_payload(result.stdout)
    _write_marker(marker_path, options, payload)
    return marker_path


def _marker_path(options: SafetyOptions) -> Path:
    key = f"{options.model_id}@{options.revision or 'main'}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return options.cache_dir / f"{digest}.json"


def _write_marker(marker_path: Path, options: SafetyOptions, payload: dict[str, object]) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker = {
        "model_id": options.model_id,
        "revision": options.revision,
        "lima_instance": options.lima_instance,
        "payload": payload,
    }
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")


def _parse_payload(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {"raw_output": stdout[-2000:]}


def _verification_script(model_id: str, revision: str | None) -> str:
    unsafe_suffixes = ", ".join(repr(suffix) for suffix in UNSAFE_MODEL_SUFFIXES)
    return textwrap.dedent(
        f"""
        import json
        import os
        from huggingface_hub import HfApi, snapshot_download
        from safetensors import safe_open

        model_id = {model_id!r}
        revision = {revision!r}
        unsafe_suffixes = ({unsafe_suffixes},)

        api = HfApi()
        info = api.model_info(model_id, revision=revision, files_metadata=True)
        files = [s.rfilename for s in info.siblings]
        unsafe = [name for name in files if name.lower().endswith(unsafe_suffixes)]
        if unsafe:
            raise SystemExit("Refusing model with unsafe serialized weights: " + ", ".join(unsafe))

        safetensor_files = [name for name in files if name.lower().endswith(".safetensors")]
        if not safetensor_files:
            raise SystemExit("Model does not publish safetensors weights.")

        cache_dir = os.path.expanduser("~/libreimage-model-safety/huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        local_dir = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.md",
                "*.model",
                "*.safetensors",
                "*.yaml",
                "*.yml",
            ],
            ignore_patterns=[
                "*.bin",
                "*.ckpt",
                "*.pt",
                "*.pth",
                "*.pkl",
                "*.pickle",
            ],
            local_files_only=False,
        )
        checked = []
        for relative_path in safetensor_files:
            path = os.path.join(local_dir, relative_path)
            with safe_open(path, framework="numpy", device="cpu") as handle:
                checked.append({{"file": relative_path, "tensors": len(handle.keys())}})

        print(json.dumps({{"safe_tensors": checked, "file_count": len(files)}}))
        """
    ).strip()


def quoted_lima_setup_hint(instance: str = DEFAULT_LIMA_INSTANCE) -> str:
    return " ".join(shlex.quote(part) for part in ["limactl", "start", instance])


def _lima_python_command() -> str:
    return textwrap.dedent(
        """
        set -euo pipefail
        tmp_dir="$(mktemp -d -t libreimage-safety.XXXXXX)"
        cleanup() { rm -rf "$tmp_dir"; }
        trap cleanup EXIT

        python3 -m venv "$tmp_dir/venv"
        "$tmp_dir/venv/bin/python" -m pip install --quiet --no-cache-dir --upgrade pip
        "$tmp_dir/venv/bin/python" -m pip install --quiet --no-cache-dir huggingface-hub safetensors numpy
        "$tmp_dir/venv/bin/python" -
        """
    ).strip()
