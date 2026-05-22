from __future__ import annotations

import argparse
from pathlib import Path

from libreimage.inpaint import DEFAULT_MODEL_ID, InpaintOptions, LocalInpainter
from libreimage.paths import default_output_path, require_image_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="libre",
        description="Run local SDXL inpainting with an explicit user-supplied mask.",
    )
    parser.add_argument("image", nargs="?", help="Image to edit. Without --mask, starts the web UI.")
    parser.add_argument("--mask", help="Mask image. White pixels are inpainted; black pixels are kept.")
    parser.add_argument("--output", "-o", help="Output path. Defaults to <image>_libre.<ext>.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Diffusers model id.")
    parser.add_argument("--revision", default=None, help="Optional model revision.")
    parser.add_argument("--prompt", default=InpaintOptions.prompt, help="Inpainting prompt.")
    parser.add_argument("--negative-prompt", default=InpaintOptions.negative_prompt)
    parser.add_argument("--steps", type=int, default=InpaintOptions.steps)
    parser.add_argument("--guidance-scale", type=float, default=InpaintOptions.guidance_scale)
    parser.add_argument("--strength", type=float, default=InpaintOptions.strength)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--lima-instance", default="default")
    parser.add_argument(
        "--skip-lima-safety",
        action="store_true",
        help="Bypass first-load Lima verification for already trusted local models.",
    )
    parser.add_argument("--web", action="store_true", help="Start the browser mask editor.")
    parser.add_argument("--host", default="127.0.0.1", help="Web UI host when starting the browser editor.")
    parser.add_argument("--port", type=int, default=7860, help="Web UI port when starting the browser editor.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.web or not args.mask:
        from libreimage.web import main as web_main

        web_args = ["--host", args.host, "--port", str(args.port)]
        if args.image:
            web_args.append(str(require_image_path(args.image)))
        return web_main(web_args)

    image_path = require_image_path(args.image)
    mask_path = Path(args.mask).expanduser().resolve()
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask does not exist: {mask_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(image_path)
    options = InpaintOptions(
        model_id=args.model,
        revision=args.revision,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        seed=args.seed,
        device=args.device,
        lima_instance=args.lima_instance,
        skip_lima_safety=args.skip_lima_safety,
    )
    result = LocalInpainter(options).run(image_path, mask_path, output_path)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
