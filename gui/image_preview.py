"""
image_preview.py

A self-contained image-preview canvas widget.

Fixes the two bugs in the previous ad-hoc _show_image() helper:
  1. winfo_width() returns 1 at construction time (widget not yet
     rendered), so the image was drawn at the wrong size.
     Fix: defer the first draw with after_idle(), and re-draw on
     <Configure> (window resize) so the preview always fills the
     canvas correctly.
  2. scale = min(..., 1.0) prevented upscaling — small images stayed
     tiny.
     Fix: removed the 1.0 cap; the image now scales to FILL the
     canvas in both directions while keeping its aspect ratio.

Usage:
    preview = ImagePreview(parent, height=250)
    preview.pack(fill="x", padx=5, pady=5)
    preview.set_image(bgr_numpy_array)   # call whenever you have a new image
    preview.clear()                       # go back to grey placeholder
"""

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np


class ImagePreview(tk.Canvas):
    """Canvas that displays a BGR numpy image, always filling the widget."""

    def __init__(self, parent, height=250, bg="#888888", **kwargs):
        super().__init__(parent, bg=bg, height=height,
                         highlightthickness=0, **kwargs)
        self._bgr_image = None   # the raw source (never mutated)
        self._photo = None       # keep a reference so GC doesn't kill it
        self._draw_pending = False

        # Redraw whenever the canvas is resized (window resize, maximise…)
        self.bind("<Configure>", self._on_resize)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_image(self, bgr_image: np.ndarray):
        """Display a new BGR numpy array. Pass None to clear."""
        self._bgr_image = bgr_image
        self._schedule_draw()

    def clear(self):
        self._bgr_image = None
        self.delete("all")
        self._photo = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _on_resize(self, event):
        """Canvas resized (window resize, first-show, etc.) → redraw."""
        self._schedule_draw()

    def _schedule_draw(self):
        """
        Schedule a single redraw on the next idle cycle.
        Collapsing multiple rapid calls (e.g. configure + set_image)
        into one prevents flicker.
        """
        if not self._draw_pending:
            self._draw_pending = True
            self.after_idle(self._draw)

    def _draw(self):
        self._draw_pending = False
        self.delete("all")

        if self._bgr_image is None:
            return

        # Real canvas dimensions (valid now that we're in after_idle)
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()

        if canvas_w < 2 or canvas_h < 2:
            # Widget not yet realised — try again shortly
            self.after(50, self._schedule_draw)
            return

        img_h, img_w = self._bgr_image.shape[:2]

        # Scale to FILL the canvas while keeping aspect ratio.
        # No artificial cap at 1.0 — small images ARE upscaled.
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(int(img_w * scale), 1)
        new_h = max(int(img_h * scale), 1)

        resized = cv2.resize(self._bgr_image, (new_w, new_h),
                              interpolation=cv2.INTER_AREA if scale < 1
                              else cv2.INTER_LINEAR)

        # BGR → RGB for PIL
        pil_img = Image.fromarray(resized[..., ::-1])
        self._photo = ImageTk.PhotoImage(pil_img)

        # Centre in canvas
        x = canvas_w // 2
        y = canvas_h // 2
        self.create_image(x, y, image=self._photo, anchor="center")
