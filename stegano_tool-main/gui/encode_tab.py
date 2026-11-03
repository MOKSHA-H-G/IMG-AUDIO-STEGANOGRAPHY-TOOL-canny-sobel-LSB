"""
encode_tab.py

Encode tab:
  - Browse a cover image (any common format)
  - Image fills the preview canvas properly at any window size
  - See the embedding-suitability heatmap (Canny+Sobel+Laplacian+Variance)
  - Type / paste a secret message
  - Embed with the adaptive multi-bit CNN-guided pipeline
  - Save the stego PNG
  - Compare against baseline sequential LSB across 7 metrics

Scrolling works because every widget is inside ScrollableFrame.content,
and the imagepreview widget defers its first draw until the canvas is
fully rendered (after_idle), then redraws on every window resize.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
import threading
import time

from gui.scrollable_frame import ScrollableFrame
from gui.image_preview import ImagePreview
from core.edge_detector import EdgeDetector
from core.intelligent_pixel_selector import IntelligentPixelSelector
from core.adaptive_embedder import AdaptiveEmbedder
from core.metrics import (
    baseline_lsb_embed, full_metrics_report,
)
from models.cnn_classifier import load_pretrained
from utils.image_utils import load_image, save_image


class EncodeTab(ttk.Frame):
    def __init__(self, parent, on_complete=None):
        super().__init__(parent)
        self.on_complete = on_complete
        self.cover_image = None
        self.cover_path = None
        self.stego_image = None
        self._cnn_model = None

        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        C = scroll.content   # everything goes here

        # ── Browse row ──────────────────────────────────────────────────
        top = ttk.Frame(C)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Button(top, text="Browse Cover Image",
                   command=self.browse).pack(side="left")
        self.file_label = ttk.Label(top, text="No image loaded")
        self.file_label.pack(side="left", padx=10)

        # ── Cover preview ───────────────────────────────────────────────
        pf = ttk.LabelFrame(C, text="Cover Image Preview")
        pf.pack(fill="x", padx=10, pady=4)
        self.cover_preview = ImagePreview(pf, height=280)
        self.cover_preview.pack(fill="x", padx=5, pady=5)

        # ── Heatmap preview ─────────────────────────────────────────────
        hf = ttk.LabelFrame(C, text=(
            "Embedding Suitability Heatmap  "
            "(red/warm = textured = good location)"))
        hf.pack(fill="x", padx=10, pady=4)
        self.heatmap_preview = ImagePreview(hf, height=200)
        self.heatmap_preview.pack(fill="x", padx=5, pady=5)

        # ── Image info ──────────────────────────────────────────────────
        info_lf = ttk.LabelFrame(C, text="Image Information")
        info_lf.pack(fill="x", padx=10, pady=4)
        self.info_label = ttk.Label(info_lf, text="Capacity: N/A",
                                     wraplength=900, justify="left")
        self.info_label.pack(anchor="w", padx=8, pady=6)

        # ── Message input ────────────────────────────────────────────────
        mf = ttk.LabelFrame(C, text="Secret Message")
        mf.pack(fill="x", padx=10, pady=4)
        ttk.Label(mf, text="Enter your secret message:"
                  ).pack(anchor="w", padx=8, pady=(6, 0))
        self.message_text = tk.Text(mf, height=7, wrap="word")
        self.message_text.pack(fill="x", padx=8, pady=6)
        self.message_text.bind("<KeyRelease>", self.update_counter)
        self.counter_label = ttk.Label(mf,
                                        text="Characters: 0 | Bits needed: 0")
        self.counter_label.pack(anchor="e", padx=8, pady=(0, 6))

        # ── Buttons ──────────────────────────────────────────────────────
        bf = ttk.Frame(C)
        bf.pack(fill="x", padx=10, pady=8)
        ttk.Button(bf, text="  Embed & Save  ",
                   command=self.run_embed).pack(side="left")

        # ── Status ───────────────────────────────────────────────────────
        self.status = ttk.Label(C, text="Ready.",
                                 relief="sunken", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(4, 14))

    # ── helpers ─────────────────────────────────────────────────────────
    def _get_cnn(self):
        if self._cnn_model is None:
            self._cnn_model = load_pretrained()
        return self._cnn_model

    def _set_status(self, text):
        self.status.config(text=text)
        self.status.update_idletasks()

    # ── browsing ─────────────────────────────────────────────────────────
    def browse(self):
        path = filedialog.askopenfilename(
            title="Select Cover Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("PNG",  "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("All",  "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.cover_image = load_image(path)
            self.cover_path  = path
            name = path.replace("\\", "/").split("/")[-1]
            self.file_label.config(text=f"Loaded: {name}")

            # Show cover image — ImagePreview scales to fill canvas
            self.cover_preview.set_image(self.cover_image)

            # Build heatmap in a thread (can take ~0.5 s on large images)
            threading.Thread(target=self._build_heatmap, daemon=True).start()
            # Capacity also computed in thread
            threading.Thread(target=self.update_capacity, daemon=True).start()

            self._set_status("Image loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def _build_heatmap(self):
        try:
            ed = EdgeDetector()
            score_map = ed.get_fused_score_map(self.cover_image)
            heatmap_bgr = ed.visualize_heatmap(score_map)
            # Update canvas from the main thread
            self.after(0, lambda: self.heatmap_preview.set_image(heatmap_bgr))
        except Exception:
            pass

    def update_capacity(self):
        if self.cover_image is None:
            self.after(0, lambda: self.info_label.config(text="Capacity: N/A"))
            return
        try:
            ed  = EdgeDetector()
            sel = IntelligentPixelSelector(self.cover_image, edge_detector=ed)
            cap = sel.get_capacity_estimate()
            h, w = self.cover_image.shape[:2]
            text = (
                f"Size: {w}×{h} px  |  "
                f"Estimated adaptive capacity: "
                f"{cap['estimated_bits']:,} bits  "
                f"(≈ {cap['estimated_characters']:,} characters)  |  "
                f"High-texture pixels: {cap['high_score_pixels']:,}  "
                f"Mid-texture: {cap['mid_score_pixels']:,}"
            )
            self.after(0, lambda: self.info_label.config(text=text))
        except Exception as e:
            self.after(0, lambda: self.info_label.config(
                text=f"Could not estimate capacity: {e}"))

    def update_counter(self, event=None):
        msg   = self.message_text.get("1.0", "end-1c")
        chars = len(msg)
        bits  = chars * 8 + 32 + 16   # + header + end-marker
        self.counter_label.config(
            text=f"Characters: {chars} | Bits needed: {bits}")

    # ── embedding ────────────────────────────────────────────────────────
    def run_embed(self):
        if self.cover_image is None:
            messagebox.showwarning("Warning", "Please select a cover image first.")
            return
        message = self.message_text.get("1.0", "end-1c")
        if not message:
            messagebox.showwarning("Warning", "Please enter a message to embed.")
            return

        self._set_status("Embedding… please wait")

        def worker():
            try:
                t0 = time.time()
                ed  = EdgeDetector()
                cnn = self._get_cnn()
                sel = IntelligentPixelSelector(
                    self.cover_image, edge_detector=ed, cnn_model=cnn)
                emb = AdaptiveEmbedder(sel)

                self.stego_image = emb.embed(self.cover_image, message)
                elapsed = (time.time() - t0) * 1000

                save_path = filedialog.asksaveasfilename(
                    title="Save Stego Image",
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                )
                if not save_path:
                    self.after(0, lambda: self._set_status(
                        "Embedding cancelled — image not saved."))
                    return

                save_image(self.stego_image, save_path)
                report = full_metrics_report(self.cover_image, self.stego_image)

                status = (
                    f"✓ Embedded!  PSNR: {report['psnr']:.2f} dB  |  "
                    f"SSIM: {report['ssim']:.4f}  |  "
                    f"Time: {elapsed:.0f} ms  |  "
                    f"Saved: {save_path.replace(chr(92),'/').split('/')[-1]}"
                )
                self.after(0, lambda: self._set_status(status))

                metrics = {
                    "cover_image": self.cover_image,
                    "stego_image": self.stego_image,
                    "message":     message,
                    "save_path":   save_path,
                    "time_ms":     elapsed,
                    **report,
                }
                if self.on_complete:
                    self.after(0, lambda: self.on_complete(metrics))

            except ValueError as e:
                self.after(0, lambda: messagebox.showerror("Capacity Error", str(e)))
                self.after(0, lambda: self._set_status("Embedding failed: insufficient capacity."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Embedding failed:\n{e}"))
                self.after(0, lambda: self._set_status("Embedding failed."))

        threading.Thread(target=worker, daemon=True).start()

    # ── comparison ───────────────────────────────────────────────────────
    def run_compare(self):
        if self.cover_image is None:
            messagebox.showwarning("Warning", "Please select a cover image first.")
            return
        message = self.message_text.get("1.0", "end-1c")
        if not message:
            messagebox.showwarning("Warning", "Please enter a message to compare.")
            return

        self._set_status("Running comparison… please wait")

        def worker():
            try:
                ed  = EdgeDetector()
                cnn = self._get_cnn()
                sel = IntelligentPixelSelector(
                    self.cover_image, edge_detector=ed, cnn_model=cnn)
                emb = AdaptiveEmbedder(sel)

                proposed_stego = emb.embed(self.cover_image, message)
                baseline_stego = baseline_lsb_embed(self.cover_image, message)

                p = full_metrics_report(self.cover_image, proposed_stego)
                b = full_metrics_report(self.cover_image, baseline_stego)

                text = (
                    "PROPOSED  (Adaptive Edge + CNN):\n"
                    f"  PSNR            {p['psnr']:.2f} dB\n"
                    f"  SSIM            {p['ssim']:.6f}\n"
                    f"  MSE             {p['mse']:.4f}\n"
                    f"  Entropy         {p['entropy']:.4f}\n"
                    f"  Histogram Chi²  {p['chi2']:.2f}\n"
                    f"  RS Analysis     {p['rs']:.4f}\n"
                    f"  Detectability   {p['detectability']:.4f}\n"
                    "\n"
                    "BASELINE  (Sequential LSB):\n"
                    f"  PSNR            {b['psnr']:.2f} dB\n"
                    f"  SSIM            {b['ssim']:.6f}\n"
                    f"  MSE             {b['mse']:.4f}\n"
                    f"  Entropy         {b['entropy']:.4f}\n"
                    f"  Histogram Chi²  {b['chi2']:.2f}\n"
                    f"  RS Analysis     {b['rs']:.4f}\n"
                    f"  Detectability   {b['detectability']:.4f}\n"
                    "\n"
                    "DIFFERENCE  (proposed − baseline):\n"
                    f"  PSNR            {p['psnr']-b['psnr']:+.2f} dB\n"
                    f"  Chi² reduced    {b['chi2']-p['chi2']:+.2f}\n"
                    f"  RS reduced      {b['rs']-p['rs']:+.4f}\n"
                    f"  Detectability↓  {b['detectability']-p['detectability']:+.4f}\n"
                    "\n"
                    "Note: PSNR/SSIM are often similar between methods since\n"
                    "both flip the same number of low bits. The proposed method's\n"
                    "real advantage is in Chi², RS, and Detectability — the method\n"
                    "embeds in high-texture regions where statistical attacks fail."
                )
                self.after(0, lambda: self._set_status("Comparison complete."))
                self.after(0, lambda: messagebox.showinfo(
                    "Comparison Results", text))

            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "Error", f"Comparison failed:\n{e}"))
                self.after(0, lambda: self._set_status("Comparison failed."))

        threading.Thread(target=worker, daemon=True).start()
"""
encode_tab.py

Encode tab:
  - Browse a cover image (any common format)
  - Image fills the preview canvas properly at any window size
  - See the embedding-suitability heatmap (Canny+Sobel+Laplacian+Variance)
  - Type / paste a secret message
  - Embed with the adaptive multi-bit CNN-guided pipeline
  - Save the stego PNG
  - Compare against baseline sequential LSB across 7 metrics

Scrolling works because every widget is inside ScrollableFrame.content,
and the imagepreview widget defers its first draw until the canvas is
fully rendered (after_idle), then redraws on every window resize.


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
import threading
import time

from gui.scrollable_frame import ScrollableFrame
from gui.image_preview import ImagePreview
from core.edge_detector import EdgeDetector
from core.intelligent_pixel_selector import IntelligentPixelSelector
from core.adaptive_embedder import AdaptiveEmbedder
from core.metrics import (
    baseline_lsb_embed, full_metrics_report,
)
from models.cnn_classifier import load_pretrained
from utils.image_utils import load_image, save_image


class EncodeTab(ttk.Frame):
    def __init__(self, parent, on_complete=None):
        super().__init__(parent)
        self.on_complete = on_complete
        self.cover_image = None
        self.cover_path = None
        self.stego_image = None
        self._cnn_model = None

        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        C = scroll.content   # everything goes here

        # ── Browse row ──────────────────────────────────────────────────
        top = ttk.Frame(C)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Button(top, text="Browse Cover Image",
                   command=self.browse).pack(side="left")
        self.file_label = ttk.Label(top, text="No image loaded")
        self.file_label.pack(side="left", padx=10)

        # ── Cover preview ───────────────────────────────────────────────
        pf = ttk.LabelFrame(C, text="Cover Image Preview")
        pf.pack(fill="x", padx=10, pady=4)
        self.cover_preview = ImagePreview(pf, height=280)
        self.cover_preview.pack(fill="x", padx=5, pady=5)

        # ── Heatmap preview ─────────────────────────────────────────────
        hf = ttk.LabelFrame(C, text=(
            "Embedding Suitability Heatmap  "
            "(red/warm = textured = good location)"))
        hf.pack(fill="x", padx=10, pady=4)
        self.heatmap_preview = ImagePreview(hf, height=200)
        self.heatmap_preview.pack(fill="x", padx=5, pady=5)

        # ── Image info ──────────────────────────────────────────────────
        info_lf = ttk.LabelFrame(C, text="Image Information")
        info_lf.pack(fill="x", padx=10, pady=4)
        self.info_label = ttk.Label(info_lf, text="Capacity: N/A",
                                     wraplength=900, justify="left")
        self.info_label.pack(anchor="w", padx=8, pady=6)

        # ── Message input ────────────────────────────────────────────────
        mf = ttk.LabelFrame(C, text="Secret Message")
        mf.pack(fill="x", padx=10, pady=4)
        ttk.Label(mf, text="Enter your secret message:"
                  ).pack(anchor="w", padx=8, pady=(6, 0))
        self.message_text = tk.Text(mf, height=7, wrap="word")
        self.message_text.pack(fill="x", padx=8, pady=6)
        self.message_text.bind("<KeyRelease>", self.update_counter)
        self.counter_label = ttk.Label(mf,
                                        text="Characters: 0 | Bits needed: 0")
        self.counter_label.pack(anchor="e", padx=8, pady=(0, 6))

        # ── Buttons ──────────────────────────────────────────────────────
        bf = ttk.Frame(C)
        bf.pack(fill="x", padx=10, pady=8)
        ttk.Button(bf, text="  Embed & Save  ",
                   command=self.run_embed).pack(side="left")

        # ── Status ───────────────────────────────────────────────────────
        self.status = ttk.Label(C, text="Ready.",
                                 relief="sunken", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(4, 14))

    # ── helpers ─────────────────────────────────────────────────────────
    def _get_cnn(self):
        if self._cnn_model is None:
            self._cnn_model = load_pretrained()
        return self._cnn_model

    def _set_status(self, text):
        self.status.config(text=text)
        self.status.update_idletasks()

    # ── browsing ─────────────────────────────────────────────────────────
    def browse(self):
        path = filedialog.askopenfilename(
            title="Select Cover Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("PNG",  "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("All",  "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.cover_image = load_image(path)
            self.cover_path  = path
            name = path.replace("\\", "/").split("/")[-1]
            self.file_label.config(text=f"Loaded: {name}")

            # Show cover image — ImagePreview scales to fill canvas
            self.cover_preview.set_image(self.cover_image)

            # Build heatmap in a thread (can take ~0.5 s on large images)
            threading.Thread(target=self._build_heatmap, daemon=True).start()
            # Capacity also computed in thread
            threading.Thread(target=self.update_capacity, daemon=True).start()

            self._set_status("Image loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def _build_heatmap(self):
        try:
            ed = EdgeDetector()
            score_map = ed.get_fused_score_map(self.cover_image)
            heatmap_bgr = ed.visualize_heatmap(score_map)
            # Update canvas from the main thread
            self.after(0, lambda: self.heatmap_preview.set_image(heatmap_bgr))
        except Exception:
            pass

    def update_capacity(self):
        if self.cover_image is None:
            self.after(0, lambda: self.info_label.config(text="Capacity: N/A"))
            return
        try:
            ed  = EdgeDetector()
            sel = IntelligentPixelSelector(self.cover_image, edge_detector=ed)
            cap = sel.get_capacity_estimate()
            h, w = self.cover_image.shape[:2]
            text = (
                f"Size: {w}×{h} px  |  "
                f"Estimated adaptive capacity: "
                f"{cap['estimated_bits']:,} bits  "
                f"(≈ {cap['estimated_characters']:,} characters)  |  "
                f"High-texture pixels: {cap['high_score_pixels']:,}  "
                f"Mid-texture: {cap['mid_score_pixels']:,}"
            )
            self.after(0, lambda: self.info_label.config(text=text))
        except Exception as e:
            self.after(0, lambda: self.info_label.config(
                text=f"Could not estimate capacity: {e}"))

    def update_counter(self, event=None):
        msg   = self.message_text.get("1.0", "end-1c")
        chars = len(msg)
        bits  = chars * 8 + 32 + 16   # + header + end-marker
        self.counter_label.config(
            text=f"Characters: {chars} | Bits needed: {bits}")

    # ── embedding ────────────────────────────────────────────────────────
    def run_embed(self):
        if self.cover_image is None:
            messagebox.showwarning("Warning", "Please select a cover image first.")
            return
        message = self.message_text.get("1.0", "end-1c")
        if not message:
            messagebox.showwarning("Warning", "Please enter a message to embed.")
            return

        # Tk dialogs MUST run on the main thread (calling them from a worker
        # thread freezes / kills the window), so ask for the path first.
        save_path = filedialog.asksaveasfilename(
            title="Save Stego Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if not save_path:
            self._set_status("Embedding cancelled — image not saved.")
            return
        if not save_path.lower().endswith(".png"):
            save_path += ".png"

        self._set_status("Embedding… please wait")
        cover = self.cover_image

        def worker():
            try:
                t0 = time.time()
                ed  = EdgeDetector()
                cnn = self._get_cnn()
                sel = IntelligentPixelSelector(
                    cover, edge_detector=ed, cnn_model=cnn)
                emb = AdaptiveEmbedder(sel)

                self.stego_image = emb.embed(cover, message)
                elapsed = (time.time() - t0) * 1000

                save_image(self.stego_image, save_path)
                report = full_metrics_report(self.cover_image, self.stego_image)

                status = (
                    f"✓ Embedded!  PSNR: {report['psnr']:.2f} dB  |  "
                    f"SSIM: {report['ssim']:.4f}  |  "
                    f"Time: {elapsed:.0f} ms  |  "
                    f"Saved: {save_path.replace(chr(92),'/').split('/')[-1]}"
                )
                self.after(0, lambda: self._set_status(status))

                metrics = {
                    "cover_image": self.cover_image,
                    "stego_image": self.stego_image,
                    "message":     message,
                    "save_path":   save_path,
                    "time_ms":     elapsed,
                    **report,
                }
                if self.on_complete:
                    self.after(0, lambda: self.on_complete(metrics))

            except ValueError as e:
                err = str(e)
                self.after(0, lambda: messagebox.showerror("Capacity Error", err))
                self.after(0, lambda: self._set_status("Embedding failed: insufficient capacity."))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: messagebox.showerror("Error", f"Embedding failed:\n{err}"))
                self.after(0, lambda: self._set_status("Embedding failed."))

        threading.Thread(target=worker, daemon=True).start()

    # ── comparison ───────────────────────────────────────────────────────
    def run_compare(self):
        if self.cover_image is None:
            messagebox.showwarning("Warning", "Please select a cover image first.")
            return
        message = self.message_text.get("1.0", "end-1c")
        if not message:
            messagebox.showwarning("Warning", "Please enter a message to compare.")
            return

        self._set_status("Running comparison… please wait")

        def worker():
            try:
                ed  = EdgeDetector()
                cnn = self._get_cnn()
                sel = IntelligentPixelSelector(
                    self.cover_image, edge_detector=ed, cnn_model=cnn)
                emb = AdaptiveEmbedder(sel)

                proposed_stego = emb.embed(self.cover_image, message)
                baseline_stego = baseline_lsb_embed(self.cover_image, message)

                p = full_metrics_report(self.cover_image, proposed_stego)
                b = full_metrics_report(self.cover_image, baseline_stego)

                text = (
                    "PROPOSED  (Adaptive Edge + CNN):\n"
                    f"  PSNR            {p['psnr']:.2f} dB\n"
                    f"  SSIM            {p['ssim']:.6f}\n"
                    f"  MSE             {p['mse']:.4f}\n"
                    f"  Entropy         {p['entropy']:.4f}\n"
                    f"  Histogram Chi²  {p['chi2']:.2f}\n"
                    f"  RS Analysis     {p['rs']:.4f}\n"
                    f"  Detectability   {p['detectability']:.4f}\n"
                    "\n"
                    "BASELINE  (Sequential LSB):\n"
                    f"  PSNR            {b['psnr']:.2f} dB\n"
                    f"  SSIM            {b['ssim']:.6f}\n"
                    f"  MSE             {b['mse']:.4f}\n"
                    f"  Entropy         {b['entropy']:.4f}\n"
                    f"  Histogram Chi²  {b['chi2']:.2f}\n"
                    f"  RS Analysis     {b['rs']:.4f}\n"
                    f"  Detectability   {b['detectability']:.4f}\n"
                    "\n"
                    "DIFFERENCE  (proposed − baseline):\n"
                    f"  PSNR            {p['psnr']-b['psnr']:+.2f} dB\n"
                    f"  Chi² reduced    {b['chi2']-p['chi2']:+.2f}\n"
                    f"  RS reduced      {b['rs']-p['rs']:+.4f}\n"
                    f"  Detectability↓  {b['detectability']-p['detectability']:+.4f}\n"
                    "\n"
                    "Note: PSNR/SSIM are often similar between methods since\n"
                    "both flip the same number of low bits. The proposed method's\n"
                    "real advantage is in Chi², RS, and Detectability — the method\n"
                    "embeds in high-texture regions where statistical attacks fail."
                )
                self.after(0, lambda: self._set_status("Comparison complete."))
                self.after(0, lambda: messagebox.showinfo(
                    "Comparison Results", text))

            except Exception as e:
                err = str(e)
                self.after(0, lambda: messagebox.showerror(
                    "Error", f"Comparison failed:\n{err}"))
                self.after(0, lambda: self._set_status("Comparison failed."))

        threading.Thread(target=worker, daemon=True).start()
"""