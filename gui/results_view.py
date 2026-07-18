"""
results_view.py

Results & Metrics tab: populated automatically after a successful embed.
Shows side-by-side cover/stego previews and a full 7-metric report.
"""

import tkinter as tk
from tkinter import ttk

from gui.scrollable_frame import ScrollableFrame
from gui.image_preview import ImagePreview


class ResultsView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        C = scroll.content

        ttk.Label(C, text="Embedding Results & Metrics",
                   font=("Arial", 14, "bold")).pack(pady=(14, 8))

        # ── Side-by-side image previews ──────────────────────────────────
        img_row = ttk.Frame(C)
        img_row.pack(fill="x", padx=10, pady=4)
        img_row.columnconfigure(0, weight=1)
        img_row.columnconfigure(1, weight=1)

        lf = ttk.LabelFrame(img_row, text="Cover Image")
        lf.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.cover_preview = ImagePreview(lf, height=270)
        self.cover_preview.pack(fill="x", padx=5, pady=5)

        rf = ttk.LabelFrame(img_row, text="Stego Image")
        rf.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.stego_preview = ImagePreview(rf, height=270)
        self.stego_preview.pack(fill="x", padx=5, pady=5)

        # ── Metrics report ───────────────────────────────────────────────
        mf = ttk.LabelFrame(C, text="Metrics Report")
        mf.pack(fill="x", padx=10, pady=8)

        self.metrics_text = tk.Text(mf, height=20, wrap="word",
                                     state="disabled", font=("Courier", 10))
        sb = ttk.Scrollbar(mf, command=self.metrics_text.yview)
        self.metrics_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.metrics_text.pack(fill="both", expand=True, padx=8, pady=8)

        # ── Message preview ──────────────────────────────────────────────
        pmf = ttk.LabelFrame(C, text="Embedded Message (preview)")
        pmf.pack(fill="x", padx=10, pady=(0, 16))
        self.msg_preview = tk.Text(pmf, height=5, wrap="word",
                                    state="disabled")
        self.msg_preview.pack(fill="x", padx=8, pady=8)

    # ── public ──────────────────────────────────────────────────────────
    def update_metrics(self, metrics: dict):
        if metrics.get("cover_image") is not None:
            self.cover_preview.set_image(metrics["cover_image"])
        if metrics.get("stego_image") is not None:
            self.stego_preview.set_image(metrics["stego_image"])

        p = metrics
        text = f"""
┌─────────────────────────────────────────────────────────┐
│              EMBEDDING RESULTS & METRICS                │
└─────────────────────────────────────────────────────────┘

QUALITY / IMPERCEPTIBILITY
  PSNR              {p.get('psnr', 0):.2f} dB
                    (> 50 dB = excellent, > 40 dB = good)
  SSIM              {p.get('ssim', 0):.6f}
                    (1.0 = identical, > 0.999 = excellent)
  MSE               {p.get('mse', 0):.4f}
                    (0.0 = identical, lower is better)

STEGANALYSIS RESISTANCE
  Entropy           {p.get('entropy', 0):.4f} bits
  Histogram Chi²    {p.get('chi2', 0):.4f}
                    (lower = histogram more similar to cover)
  RS Analysis       {p.get('rs', 0):.4f}
                    (< 0.05 = highly resistant to RS attack,0.05-0.1=moderate, > 0.1 = weak)
  Detectability     {p.get('detectability', 0):.4f}
                    (0.0 = undetectable, 1.0 = obvious)

PERFORMANCE
  Embedding time    {p.get('time_ms', 0):.1f} ms
  Message length    {len(p.get('message', ''))} characters
  Saved to          {p.get('save_path', 'N/A')}


""".strip()

        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", text)
        self.metrics_text.config(state="disabled")

        msg = p.get("message", "")
        preview = msg[:400] + ("…" if len(msg) > 400 else "")
        self.msg_preview.config(state="normal")
        self.msg_preview.delete("1.0", "end")
        self.msg_preview.insert("1.0", preview)
        self.msg_preview.config(state="disabled")
