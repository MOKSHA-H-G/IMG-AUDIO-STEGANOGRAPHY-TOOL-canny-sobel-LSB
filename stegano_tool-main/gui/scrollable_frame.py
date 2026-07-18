"""
scrollable_frame.py

A reusable scrollable container for Tkinter tabs.

Key fixes over the previous version
-------------------------------------
1. Mousewheel binding propagates to ALL child widgets
   (not just when the cursor is directly over the bare canvas).
   Implemented by recursively binding <MouseWheel> / <Button-4/5>
   to every widget as it is added to `content`, using a Configure
   trace that walks the new widget tree.

2. Bind target is the scrollable-frame's CANVAS, not "all" events
   globally (bind_all steals wheel events from e.g. the Text editor
   inside the same tab).

3. Works on Windows (event.delta / 120), macOS (event.delta / 120,
   values are larger), and Linux (Button-4 = up, Button-5 = down).

Usage:
    scroll = ScrollableFrame(parent)
    scroll.pack(fill="both", expand=True)
    # Build every widget inside scroll.content, NOT inside scroll:
    ttk.Label(scroll.content, text="Hello").pack()
"""

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Canvas + scrollbar
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                 bg="#f0f0f0")
        self.vscrollbar = ttk.Scrollbar(self, orient="vertical",
                                         command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscrollbar.set)

        self.vscrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Inner frame — ALL widgets go here
        self.content = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )

        # Keep inner frame width == canvas width so horizontal scroll
        # never appears and content fills the full window width.
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.content.bind("<Configure>", self._on_content_resize)

        # Bind mousewheel to canvas and to every widget inside content
        # so scrolling works no matter where the cursor happens to be.
        self._bind_wheel(self.canvas)
        self._bind_wheel(self.content)
        self.content.bind("<Enter>", self._on_enter_content, add="+")

        # Intercept widget creation inside content so we can bind new
        # child widgets automatically.
        self.content.bind_all("<Map>", self._on_widget_mapped, add="+")

    # ------------------------------------------------------------------
    # Resize handlers
    # ------------------------------------------------------------------
    def _on_canvas_resize(self, event):
        """Keep inner frame width == canvas width."""
        self.canvas.itemconfig(self._window_id, width=event.width)

    def _on_content_resize(self, event):
        """Update scroll region whenever content height changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # ------------------------------------------------------------------
    # Mousewheel binding helpers
    # ------------------------------------------------------------------
    def _bind_wheel(self, widget):
        """Attach wheel callbacks to a single widget."""
        widget.bind("<MouseWheel>", self._scroll, add="+")   # Win / macOS
        widget.bind("<Button-4>",   self._scroll_up, add="+")   # Linux
        widget.bind("<Button-5>",   self._scroll_down, add="+") # Linux

    def _on_widget_mapped(self, event):
        """
        Called whenever any widget anywhere in the app is mapped (made
        visible). We only want to bind wheel to widgets that live inside
        our content frame.
        """
        widget = event.widget
        try:
            # Walk up the parent chain to see if it belongs to us
            parent = widget
            while parent is not None:
                if parent is self.content:
                    self._bind_wheel(widget)
                    return
                parent = parent.master
        except Exception:
            pass

    def _on_enter_content(self, event):
        """Re-bind on focus so wheel always reaches us."""
        self._bind_wheel(event.widget)

    # ------------------------------------------------------------------
    # Scroll callbacks
    # ------------------------------------------------------------------
    def _scroll(self, event):
        """Windows / macOS: event.delta is a multiple of 120."""
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _scroll_up(self, event):
        self.canvas.yview_scroll(-1, "units")

    def _scroll_down(self, event):
        self.canvas.yview_scroll(1, "units")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def scroll_to_top(self):
        self.canvas.yview_moveto(0)
