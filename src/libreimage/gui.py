from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from libreimage.images import load_rgb
from libreimage.inpaint import DEFAULT_MODEL_ID, InpaintOptions, LocalInpainter
from libreimage.paths import default_output_path, require_image_path


class MaskEditor(tk.Tk):
    def __init__(self, image_path: Path | None = None) -> None:
        super().__init__()
        self.title("LibreImage")
        self.geometry("1180x820")
        self.minsize(760, 520)

        self.image_path: Path | None = None
        self.image: Image.Image | None = None
        self.mask: Image.Image | None = None
        self.preview: ImageTk.PhotoImage | None = None
        self.display_box = (0, 0, 1, 1)
        self.history: list[Image.Image] = []
        self.painting = False
        self.last_point: tuple[int, int] | None = None
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self.brush_size = tk.IntVar(value=48)
        self.mode = tk.StringVar(value="paint")
        self.prompt = tk.StringVar(value=InpaintOptions.prompt)
        self.steps = tk.IntVar(value=InpaintOptions.steps)
        self.guidance = tk.DoubleVar(value=InpaintOptions.guidance_scale)
        self.strength = tk.DoubleVar(value=InpaintOptions.strength)
        self.model_id = tk.StringVar(value=DEFAULT_MODEL_ID)
        self.skip_lima = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Open an image to begin.")

        self._build_ui()
        self.bind("<Configure>", self._on_resize)
        self.after(120, self._poll_worker)

        if image_path is not None:
            self.open_image(image_path)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=(8, 8, 8, 4))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open", command=self._choose_image).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Clear", command=self.clear_mask).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Radiobutton(toolbar, text="Brush", variable=self.mode, value="paint").pack(side=tk.LEFT, padx=(16, 0))
        ttk.Radiobutton(toolbar, text="Eraser", variable=self.mode, value="erase").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(toolbar, text="Size").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Scale(toolbar, from_=6, to=180, variable=self.brush_size, orient=tk.HORIZONTAL, length=150).pack(
            side=tk.LEFT
        )
        ttk.Button(toolbar, text="Inpaint", command=self.run_inpaint).pack(side=tk.RIGHT)

        settings = ttk.Frame(self, padding=(8, 0, 8, 8))
        settings.pack(side=tk.TOP, fill=tk.X)
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)

        ttk.Label(settings, text="Prompt").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.prompt).grid(row=0, column=1, sticky="ew", padx=(6, 12))
        ttk.Label(settings, text="Model").grid(row=0, column=2, sticky="w")
        ttk.Entry(settings, textvariable=self.model_id).grid(row=0, column=3, sticky="ew", padx=(6, 0))

        numeric = ttk.Frame(settings)
        numeric.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Label(numeric, text="Steps").pack(side=tk.LEFT)
        ttk.Spinbox(numeric, from_=1, to=100, textvariable=self.steps, width=5).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(numeric, text="Guidance").pack(side=tk.LEFT)
        ttk.Spinbox(numeric, from_=1.0, to=20.0, increment=0.5, textvariable=self.guidance, width=5).pack(
            side=tk.LEFT, padx=(4, 14)
        )
        ttk.Label(numeric, text="Strength").pack(side=tk.LEFT)
        ttk.Spinbox(numeric, from_=0.1, to=1.0, increment=0.05, textvariable=self.strength, width=5).pack(
            side=tk.LEFT, padx=(4, 14)
        )
        ttk.Checkbutton(numeric, text="Skip Lima safety", variable=self.skip_lima).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self, bg="#202124", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._continue_stroke)
        self.canvas.bind("<ButtonRelease-1>", self._end_stroke)

        statusbar = ttk.Frame(self, padding=(8, 5))
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(statusbar, textvariable=self.status, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _choose_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.open_image(Path(filename))

    def open_image(self, path: Path) -> None:
        try:
            self.image_path = require_image_path(path)
            self.image = load_rgb(self.image_path)
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))
            return

        self.mask = Image.new("L", self.image.size, 0)
        self.history.clear()
        self.status.set(f"Loaded {self.image_path.name}. Paint the areas to replace.")
        self._redraw()

    def undo(self) -> None:
        if self.history:
            self.mask = self.history.pop()
            self._redraw()

    def clear_mask(self) -> None:
        if self.image is None or self.mask is None:
            return
        self._save_history()
        self.mask = Image.new("L", self.image.size, 0)
        self._redraw()

    def run_inpaint(self) -> None:
        if self.image is None or self.mask is None or self.image_path is None:
            messagebox.showinfo("No image", "Open an image before running inpaint.")
            return
        if self.mask.getbbox() is None:
            messagebox.showinfo("No mask", "Paint a mask before running inpaint.")
            return

        output_path = default_output_path(self.image_path)
        options = InpaintOptions(
            model_id=self.model_id.get().strip() or DEFAULT_MODEL_ID,
            prompt=self.prompt.get().strip() or InpaintOptions.prompt,
            steps=max(1, int(self.steps.get())),
            guidance_scale=float(self.guidance.get()),
            strength=float(self.strength.get()),
            skip_lima_safety=bool(self.skip_lima.get()),
        )
        image = self.image.copy()
        mask = self.mask.copy()
        self.status.set("Running inpaint. Model loading can take a while on first use.")
        self._set_controls_state(tk.DISABLED)

        thread = threading.Thread(
            target=self._inpaint_worker,
            args=(options, image, mask, output_path),
            daemon=True,
        )
        thread.start()

    def _inpaint_worker(
        self,
        options: InpaintOptions,
        image: Image.Image,
        mask: Image.Image,
        output_path: Path,
    ) -> None:
        try:
            result = LocalInpainter(options).run_images(image, mask)
            result.save(output_path)
        except Exception as exc:
            self.worker_queue.put(("error", exc))
            return
        self.worker_queue.put(("done", output_path))

    def _poll_worker(self) -> None:
        try:
            kind, payload = self.worker_queue.get_nowait()
        except queue.Empty:
            self.after(120, self._poll_worker)
            return

        self._set_controls_state(tk.NORMAL)
        if kind == "done":
            path = Path(payload)
            self.status.set(f"Saved {path.name}")
            messagebox.showinfo("Inpaint complete", f"Saved:\n{path}")
        else:
            self.status.set("Inpaint failed.")
            messagebox.showerror("Inpaint failed", str(payload))
        self.after(120, self._poll_worker)

    def _set_controls_state(self, state: str) -> None:
        for child in self.winfo_children():
            self._set_child_state(child, state)
        self.canvas.configure(cursor="watch" if state == tk.DISABLED else "")

    def _set_child_state(self, widget: tk.Widget, state: str) -> None:
        if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Spinbox, ttk.Checkbutton, ttk.Radiobutton, ttk.Scale)):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
        for child in widget.winfo_children():
            self._set_child_state(child, state)

    def _start_stroke(self, event: tk.Event) -> None:
        if self.image is None or self.mask is None:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        self._save_history()
        self.painting = True
        self.last_point = point
        self._draw_at(point, point)

    def _continue_stroke(self, event: tk.Event) -> None:
        if not self.painting or self.last_point is None:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        self._draw_at(self.last_point, point)
        self.last_point = point

    def _end_stroke(self, _event: tk.Event) -> None:
        self.painting = False
        self.last_point = None

    def _draw_at(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        if self.mask is None:
            return
        draw = ImageDraw.Draw(self.mask)
        value = 255 if self.mode.get() == "paint" else 0
        width = int(self.brush_size.get())
        draw.line([start, end], fill=value, width=width, joint="curve")
        radius = width // 2
        draw.ellipse((end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius), fill=value)
        self._redraw()

    def _save_history(self) -> None:
        if self.mask is None:
            return
        self.history.append(self.mask.copy())
        if len(self.history) > 25:
            self.history.pop(0)

    def _canvas_to_image(self, x: int, y: int) -> tuple[int, int] | None:
        if self.image is None:
            return None
        left, top, width, height = self.display_box
        if x < left or y < top or x > left + width or y > top + height:
            return None
        image_x = int((x - left) * self.image.width / width)
        image_y = int((y - top) * self.image.height / height)
        return (
            min(max(image_x, 0), self.image.width - 1),
            min(max(image_y, 0), self.image.height - 1),
        )

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget == self:
            self.after_idle(self._redraw)

    def _redraw(self) -> None:
        if self.image is None or self.mask is None:
            self.canvas.delete("all")
            return

        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        scale = min(canvas_width / self.image.width, canvas_height / self.image.height, 1.0)
        display_size = (max(1, int(self.image.width * scale)), max(1, int(self.image.height * scale)))
        left = (canvas_width - display_size[0]) // 2
        top = (canvas_height - display_size[1]) // 2
        self.display_box = (left, top, display_size[0], display_size[1])

        preview = self.image.resize(display_size, Image.Resampling.LANCZOS)
        mask = self.mask.resize(display_size, Image.Resampling.NEAREST)
        overlay = Image.new("RGBA", display_size, (255, 72, 72, 0))
        overlay.putalpha(mask.point(lambda value: 92 if value else 0))
        composed = Image.alpha_composite(preview.convert("RGBA"), overlay)

        self.preview = ImageTk.PhotoImage(composed)
        self.canvas.delete("all")
        self.canvas.create_image(left, top, image=self.preview, anchor=tk.NW)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="libre-gui", description="Open the LibreImage mask editor.")
    parser.add_argument("image", nargs="?", help="Optional image to open.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = Path(args.image).expanduser().resolve() if args.image else None
    app = MaskEditor(image_path)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
