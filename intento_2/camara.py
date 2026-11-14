import tkinter as tk
import cv2
import mss
import numpy as np
from PIL import Image, ImageTk


def seleccionar_region_pantalla():
    """
    Captura el escritorio actual y lo muestra en pantalla completa sin oscurecer,
    permitiendo dibujar el recuadro rojo con precisión sobre el simulador.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Monitor principal
        sct_img = sct.grab(monitor)
        img_bg = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.config(cursor="cross")

    tk_img = ImageTk.PhotoImage(img_bg)
    canvas = tk.Canvas(root, width=img_bg.width, height=img_bg.height, highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

    coords = {}

    def on_press(event):
        coords["x1"] = event.x
        coords["y1"] = event.y
        coords["rect"] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#ff0000", width=4
        )

    def on_drag(event):
        canvas.coords(coords["rect"], coords["x1"], coords["y1"], event.x, event.y)

    def on_release(event):
        coords["x2"] = event.x
        coords["y2"] = event.y
        root.destroy()

    # Presionar ESC cancela la selección
    root.bind("<Escape>", lambda e: root.destroy())
    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)

    root.mainloop()

    if "x1" in coords and "x2" in coords:
        x1, x2 = min(coords["x1"], coords["x2"]), max(coords["x1"], coords["x2"])
        y1, y2 = min(coords["y1"], coords["y2"]), max(coords["y1"], coords["y2"])
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            return {"left": int(x1), "top": int(y1), "width": int(x2 - x1), "height": int(y2 - y1)}
    return None


def capturar_frame_vivo(region: dict) -> np.ndarray:
    """Captura un fotograma BGR en tiempo real de la región definida."""
    with mss.mss() as sct:
        sct_img = sct.grab(region)
        frame = np.array(sct_img)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)