"""
decode_tab.py

Decode tab: browse a stego PNG, extract the hidden message using the
same adaptive pipeline as the encoder.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

from gui.scrollable_frame import ScrollableFrame
from gui.image_preview import ImagePreview
from core.edge_detector import EdgeDetector
from core.intelligent_pixel_selector import IntelligentPixelSelector
from core.adaptive_embedder import AdaptiveExtractor
from models.cnn_classifier import load_pretrained
from utils.image_utils import load_image


class DecodeTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.stego_image = None
        self._cnn_model  = None

        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        C = scroll.content

        # ── Browse ───────────────────────────────────────────────────────
        top = ttk.Frame(C)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Button(top, text="Browse Stego Image",
                   command=self.browse).pack(side="left")
        self.file_label = ttk.Label(top, text="No image loaded")
        self.file_label.pack(side="left", padx=10)

        # ── Preview ──────────────────────────────────────────────────────
        pf = ttk.LabelFrame(C, text="Stego Image Preview")
        pf.pack(fill="x", padx=10, pady=4)
        self.preview = ImagePreview(pf, height=280)
        self.preview.pack(fill="x", padx=5, pady=5)

        # ── Extract button ───────────────────────────────────────────────
        bf = ttk.Frame(C)
        bf.pack(fill="x", padx=10, pady=8)
        ttk.Button(bf, text="  Extract Message  ",
                   command=self.run_extract).pack(side="left")

        # ── Result ───────────────────────────────────────────────────────
        mf = ttk.LabelFrame(C, text="Extracted Message")
        mf.pack(fill="x", padx=10, pady=4)
        self.message_text = tk.Text(mf, height=10, wrap="word",
                                     state="disabled")
        self.message_text.pack(fill="x", padx=8, pady=6)

        cf = ttk.Frame(mf)
        cf.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(cf, text="Copy to Clipboard",
                   command=self.copy_to_clipboard).pack(side="left")

        # ── Status ───────────────────────────────────────────────────────
        self.status = ttk.Label(C, text="Ready.",
                                 relief="sunken", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(4, 14))

    # ── helpers ──────────────────────────────────────────────────────────
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
            title="Select Stego Image",
            filetypes=[
                ("PNG files",   "*.png"),
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files",   "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.stego_image = load_image(path)
            name = path.replace("\\", "/").split("/")[-1]
            self.file_label.config(text=f"Loaded: {name}")
            self.preview.set_image(self.stego_image)
            self._set_status("Image loaded. Click 'Extract Message' to continue.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    # ── extraction ───────────────────────────────────────────────────────
    def run_extract(self):
        if self.stego_image is None:
            messagebox.showwarning("Warning", "Please select a stego image first.")
            return

        self._set_status("Extracting… please wait")

        def worker():
            try:
                ed  = EdgeDetector()
                cnn = self._get_cnn()
                sel = IntelligentPixelSelector(
                    self.stego_image, edge_detector=ed, cnn_model=cnn)
                ext = AdaptiveExtractor(sel)

                message = ext.extract(self.stego_image)

                def show():
                    self.message_text.config(state="normal")
                    self.message_text.delete("1.0", "end")
                    self.message_text.insert("1.0", message)
                    self.message_text.config(state="disabled")
                    self._set_status(
                        f"✓ Extraction successful!  ({len(message)} characters)")

                self.after(0, show)

            except ValueError as e:
                self.after(0, lambda: messagebox.showerror(
                    "Extraction Failed", str(e)))
                self.after(0, lambda: self._set_status("Extraction failed."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "Error", f"Extraction failed:\n{e}"))
                self.after(0, lambda: self._set_status("Extraction error."))

        threading.Thread(target=worker, daemon=True).start()

    def copy_to_clipboard(self):
        self.message_text.config(state="normal")
        msg = self.message_text.get("1.0", "end-1c")
        self.message_text.config(state="disabled")
        if not msg:
            messagebox.showinfo("Info", "No message to copy.")
            return
        self.clipboard_clear()
        self.clipboard_append(msg)
        messagebox.showinfo("Success", "Message copied to clipboard!")
