"""
app.py  —  Main application window
"""

import tkinter as tk
from tkinter import ttk
from gui.encode_tab   import EncodeTab
from gui.decode_tab   import DecodeTab
from gui.results_view import ResultsView
from gui.audio_tab    import AudioTab


class StegoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(
            "Robust Steganography System  —  Adaptive Edge + CNN Guided Embedding")
        self.geometry("1150x820")
        self.minsize(750, 540)

        nb = ttk.Notebook(self)
        nb.pack(expand=True, fill="both", padx=5, pady=5)
        self.nb = nb

        self.encode_tab   = EncodeTab(nb, on_complete=self._on_encode_done)
        self.decode_tab   = DecodeTab(nb)
        self.results_view = ResultsView(nb)
        self.audio_tab    = AudioTab(nb)

        nb.add(self.encode_tab,   text="  Encode  ")
        nb.add(self.decode_tab,   text="  Decode  ")
        nb.add(self.results_view, text="  Metrics & Results  ")
        nb.add(self.audio_tab,    text="  Audio Steganography  ")

    def _on_encode_done(self, metrics: dict):
        self.results_view.update_metrics(metrics)
        self.nb.select(2)


def launch():
    app = StegoApp()
    app.mainloop()


if __name__ == "__main__":
    launch()
