"""
audio_tab.py

Audio Steganography tab: a sub-notebook with its own Encode / Decode
pages, mirroring the image Encode/Decode tabs but for WAV audio using
straightforward sequential 1-bit-per-sample LSB embedding.

Kept in a single file (unlike the image encode/decode split) since the
audio pipeline has no CNN/heatmap step and each page is short.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import wave

from gui.scrollable_frame import ScrollableFrame
from core.audio_embedder import AudioEmbedder, AudioExtractor
from core.audio_metrics import audio_metrics_report
from utils.audio_utils import load_audio, save_audio, capacity_bits

_WAV_FILETYPES = [("WAV files", "*.wav"), ("All files", "*.*")]


def _basename(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1]


class AudioEncodePage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.cover_samples = None
        self.cover_params = None
        self.cover_path = None
        self.stego_samples = None

        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        C = scroll.content

        top = ttk.Frame(C)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Button(top, text="Browse Cover Audio (WAV)",
                   command=self.browse).pack(side="left")
        self.file_label = ttk.Label(top, text="No audio loaded")
        self.file_label.pack(side="left", padx=10)

        info_lf = ttk.LabelFrame(C, text="Audio Information")
        info_lf.pack(fill="x", padx=10, pady=4)
        self.info_label = ttk.Label(info_lf, text="Capacity: N/A",
                                     wraplength=900, justify="left")
        self.info_label.pack(anchor="w", padx=8, pady=6)

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

        bf = ttk.Frame(C)
        bf.pack(fill="x", padx=10, pady=8)
        ttk.Button(bf, text="  Embed & Save  ",
                   command=self.run_embed).pack(side="left")

        rf = ttk.LabelFrame(C, text="Embedding Quality")
        rf.pack(fill="x", padx=10, pady=4)
        self.result_label = ttk.Label(rf, text="No embedding performed yet.",
                                       wraplength=900, justify="left")
        self.result_label.pack(anchor="w", padx=8, pady=6)

        self.status = ttk.Label(C, text="Ready.", relief="sunken", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(4, 14))

    def _set_status(self, text):
        self.status.config(text=text)
        self.status.update_idletasks()

    def browse(self):
        path = filedialog.askopenfilename(
            title="Select Cover Audio", filetypes=_WAV_FILETYPES)
        if not path:
            return
        try:
            self.cover_samples, self.cover_params = load_audio(path)
            self.cover_path = path
            self.file_label.config(text=f"Loaded: {_basename(path)}")

            cap_bits = capacity_bits(self.cover_samples)
            p = self.cover_params
            text = (
                f"Channels: {p.nchannels}  |  Sample width: {p.sampwidth * 8}-bit  |  "
                f"Sample rate: {p.framerate} Hz  |  Duration: {p.duration_seconds():.2f}s  |  "
                f"Capacity: {cap_bits:,} bits (≈ {cap_bits // 8:,} characters)"
            )
            self.info_label.config(text=text)
            self._set_status("Audio loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio:\n{e}")

    def update_counter(self, event=None):
        msg = self.message_text.get("1.0", "end-1c")
        chars = len(msg)
        bits = chars * 8 + 32 + 16  # + header + end-marker
        self.counter_label.config(
            text=f"Characters: {chars} | Bits needed: {bits}")

    def run_embed(self):
        if self.cover_samples is None:
            messagebox.showwarning("Warning", "Please select a cover audio file first.")
            return
        message = self.message_text.get("1.0", "end-1c")
        if not message:
            messagebox.showwarning("Warning", "Please enter a message to embed.")
            return

        self._set_status("Embedding… please wait")

        def worker():
            try:
                t0 = time.time()
                emb = AudioEmbedder()
                self.stego_samples = emb.embed(self.cover_samples, message)
                elapsed = (time.time() - t0) * 1000

                save_path = filedialog.asksaveasfilename(
                    title="Save Stego Audio", defaultextension=".wav",
                    filetypes=_WAV_FILETYPES,
                )
                if not save_path:
                    self.after(0, lambda: self._set_status(
                        "Embedding cancelled — audio not saved."))
                    return

                save_audio(self.stego_samples, self.cover_params, save_path)
                report = audio_metrics_report(
                    self.cover_samples, self.stego_samples,
                    self.cover_params.sampwidth)

                status = (
                    f"✓ Embedded!  SNR: {report['snr']:.2f} dB  |  "
                    f"Time: {elapsed:.0f} ms  |  "
                    f"Saved: {_basename(save_path)}"
                )
                result_text = (
                    f"SNR: {report['snr']:.2f} dB  |  PSNR: {report['psnr']:.2f} dB  |  "
                    f"Samples changed: {report['samples_changed']:,} / "
                    f"{self.cover_samples.size:,}  |  "
                    f"Max sample delta: {report['max_abs_delta']} "
                    "(should be 1 -- only the LSB is ever touched)"
                )
                self.after(0, lambda: self._set_status(status))
                self.after(0, lambda: self.result_label.config(text=result_text))

            except ValueError as e:
                self.after(0, lambda: messagebox.showerror("Capacity Error", str(e)))
                self.after(0, lambda: self._set_status("Embedding failed: insufficient capacity."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Embedding failed:\n{e}"))
                self.after(0, lambda: self._set_status("Embedding failed."))

        threading.Thread(target=worker, daemon=True).start()


class AudioDecodePage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.stego_samples = None

        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        C = scroll.content

        top = ttk.Frame(C)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Button(top, text="Browse Stego Audio (WAV)",
                   command=self.browse).pack(side="left")
        self.file_label = ttk.Label(top, text="No audio loaded")
        self.file_label.pack(side="left", padx=10)

        info_lf = ttk.LabelFrame(C, text="Audio Information")
        info_lf.pack(fill="x", padx=10, pady=4)
        self.info_label = ttk.Label(info_lf, text="N/A",
                                     wraplength=900, justify="left")
        self.info_label.pack(anchor="w", padx=8, pady=6)

        bf = ttk.Frame(C)
        bf.pack(fill="x", padx=10, pady=8)
        ttk.Button(bf, text="  Extract Message  ",
                   command=self.run_extract).pack(side="left")

        mf = ttk.LabelFrame(C, text="Extracted Message")
        mf.pack(fill="x", padx=10, pady=4)
        self.message_text = tk.Text(mf, height=10, wrap="word", state="disabled")
        self.message_text.pack(fill="x", padx=8, pady=6)

        cf = ttk.Frame(mf)
        cf.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(cf, text="Copy to Clipboard",
                   command=self.copy_to_clipboard).pack(side="left")

        self.status = ttk.Label(C, text="Ready.", relief="sunken", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(4, 14))

    def _set_status(self, text):
        self.status.config(text=text)
        self.status.update_idletasks()

    def browse(self):
        path = filedialog.askopenfilename(
            title="Select Stego Audio", filetypes=_WAV_FILETYPES)
        if not path:
            return
        try:
            self.stego_samples, params = load_audio(path)
            self.file_label.config(text=f"Loaded: {_basename(path)}")
            text = (
                f"Channels: {params.nchannels}  |  Sample width: {params.sampwidth * 8}-bit  |  "
                f"Sample rate: {params.framerate} Hz  |  Duration: {params.duration_seconds():.2f}s"
            )
            self.info_label.config(text=text)
            self._set_status("Audio loaded. Click 'Extract Message' to continue.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio:\n{e}")

    def run_extract(self):
        if self.stego_samples is None:
            messagebox.showwarning("Warning", "Please select a stego audio file first.")
            return

        self._set_status("Extracting… please wait")

        def worker():
            try:
                ext = AudioExtractor()
                message = ext.extract(self.stego_samples)

                def show():
                    self.message_text.config(state="normal")
                    self.message_text.delete("1.0", "end")
                    self.message_text.insert("1.0", message)
                    self.message_text.config(state="disabled")
                    self._set_status(
                        f"✓ Extraction successful!  ({len(message)} characters)")

                self.after(0, show)

            except ValueError as e:
                self.after(0, lambda: messagebox.showerror("Extraction Failed", str(e)))
                self.after(0, lambda: self._set_status("Extraction failed."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Extraction failed:\n{e}"))
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


class AudioTab(ttk.Frame):
    """Top-level 'Audio Steganography' tab, holding Encode/Decode sub-tabs."""

    def __init__(self, parent):
        super().__init__(parent)
        sub_nb = ttk.Notebook(self)
        sub_nb.pack(expand=True, fill="both", padx=4, pady=4)

        self.encode_page = AudioEncodePage(sub_nb)
        self.decode_page = AudioDecodePage(sub_nb)

        sub_nb.add(self.encode_page, text="  Encode  ")
        sub_nb.add(self.decode_page, text="  Decode  ")
