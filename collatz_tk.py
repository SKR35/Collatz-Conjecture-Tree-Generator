#!/usr/bin/env python3
# Collatz Conjecture Tree Generator (Tkinter)
# - Static render or animated growth
# - AnglePath-style turns (left for even, right for odd) with decay
# - Density-based coloring using custom / matplotlib colormaps + gamma
# - Save PNG
#
# Requirements:
#   pip install pillow numpy matplotlib
#
# Run:
#   python collatz_tk.py

import math, time, os, random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

#Pillow for fast blit
try:
    from PIL import Image, ImageTk, ImageDraw
except Exception as e:
    raise SystemExit("Pillow is required. Install with:  pip install pillow") from e

#Matplotlib for colormaps (no GUI)
try:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# --- Palettes (mirrors mandelbrot_tk.py) ------------------------------------
CMAPS = [
    "plasma", "viridis", "magma", "inferno", "turbo", "cividis", "twilight",
    "SoftSunset", "EarthAndSky", "Seashore", "Forest", "HotAndCold",
    "Pastel", "Grayscale"]
    
DEFAULT_CMAP = "SoftSunset"

CUSTOM_STOPS = {
    "SoftSunset":  ["#2b1055","#6a0572","#ff6f91","#ffc15e","#ffe29a"],
    "EarthAndSky": ["#1a2a6c","#28a0b0","#84ffc9","#f0f3bd","#ffd166"],
    "Seashore":    ["#001219","#005f73","#0a9396","#94d2bd","#e9d8a6"],
    "Forest":      ["#0b3d0b","#236e3c","#4caf50","#a8e6cf","#f1f8e9"],
    "HotAndCold":  ["#313695","#4575b4","#74add1","#abd9e9","#fee090","#f46d43","#d73027"],
    "Pastel":      ["#b3e5fc","#c5cae9","#e1bee7","#f8bbd0","#ffe0b2","#dcedc8"],
    "Grayscale":   ["#0a0a0a","#2f2f2f","#5e5e5e","#9a9a9a","#cccccc","#f2f2f2"],
}
def _build_lut(stops, n=1024):
    if HAVE_MPL:
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", stops, N=n)
        return (cmap(np.linspace(0,1,n))[:,:3]*255.0).astype(np.uint8)
    #Simple manual gradient fallback
    cols = [tuple(int(stops[i].lstrip("#")[j:j+2],16) for j in (0,2,4)) for i in range(len(stops))]
    lut = np.zeros((n,3),dtype=np.uint8)
    for i in range(n):
        t = i/(n-1)
        k = min(int(t*(len(cols)-1)), len(cols)-2)
        local = t*(len(cols)-1)-k
        a,b = np.array(cols[k]), np.array(cols[k+1])
        lut[i] = (a*(1-local)+b*local).astype(np.uint8)
    return lut
_CUSTOM_LUTS = {name: _build_lut(stops) for name, stops in CUSTOM_STOPS.items()}

def map_to_rgb(vals: np.ndarray, cmap_name: str, lo: float, hi: float, gamma: float = 1.35, smoothstep: bool = True) -> np.ndarray:
    """Map scalar field to RGB uint8 with soft transitions."""
    x = np.clip((vals - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    if smoothstep:
        x = x * x * (3.0 - 2.0 * x)
    if gamma and gamma > 0:
        x = np.power(x, gamma)

    if cmap_name in _CUSTOM_LUTS:
        lut = _CUSTOM_LUTS[cmap_name]
        idx = np.minimum((x * (len(lut)-1)).astype(np.int32), len(lut)-1)
        rgb = lut[idx]
    elif HAVE_MPL:
        cmap = cm.get_cmap(cmap_name)
        rgb = (cmap(x)[..., :3] * 255.0).astype(np.uint8)
    else:
        # very basic fallback
        r = (0.6 + 0.4*x) * 255
        g = (0.0 + 0.9*x) * 255
        b = (0.6 - 0.5*x) * 255
        rgb = np.dstack([r, g, b]).astype(np.uint8)
    return rgb

# --- Collatz ---------------------------------------------------------------
def collatz_sequence(n: int, max_len: int = 4096) -> List[int]:
    """Return Collatz sequence from n down to 1 (inclusive) capped by max_len."""
    out = [n]
    while n != 1 and len(out) < max_len:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3*n + 1
        out.append(n)
    return out

def sequence_to_path(seq: List[int], turn_deg: float, step0: float, decay: float) -> List[Tuple[float,float]]:
    """Convert a collatz sequence into an angle-path.
       Turn left on even, right on odd. Step size shrinks multiplicatively by 'decay' each step.
    """
    x, y = 0.0, 0.0
    ang = 0.0                 #Start pointing to the right
    step = step0
    pts = [(x, y)]
    for k, val in enumerate(seq[:-1]):  #Last term 1 contributes a final segment
        if val % 2 == 0:
            ang += math.radians(turn_deg)     #Left for even
        else:
            ang -= math.radians(turn_deg)     #Right for odd
        x += step * math.cos(ang)
        y -= step * math.sin(ang)             #Canvas y grows downward
        pts.append((x, y))
        step *= decay
    return pts

# --- Rasterization ---------------------------------------------------------
def draw_paths_to_density(paths: List[List[Tuple[float,float]]], w: int, h: int, margin: int = 24) -> np.ndarray:
    """Render many polyline paths into a density map (float array).
       We fit all coordinates in a bounding box into the canvas with margins.
    """
    #Gather all points
    if not paths:
        return np.zeros((h, w), dtype=np.float32)
    xs = np.concatenate([np.array([p[0] for p in path], dtype=np.float32) for path in paths])
    ys = np.concatenate([np.array([p[1] for p in path], dtype=np.float32) for path in paths])
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    spanx = max(1e-6, maxx - minx)
    spany = max(1e-6, maxy - miny)

    #Fit to canvas while preserving aspect (left-to-right canopy look)
    target_aspect = h / w
    box_aspect = spany / spanx
    if box_aspect > target_aspect:
        #Too tall: widen spanx to match canvas
        spanx = spany / target_aspect
        midx = (minx + maxx) * 0.5
        minx = midx - spanx/2
        maxx = midx + spanx/2
    else:
        #Too wide: increase spany
        spany = spanx * target_aspect
        midy = (miny + maxy) * 0.5
        miny = midy - spany/2
        maxy = midy + spany/2

    scale_x = (w - 2*margin) / spanx
    scale_y = (h - 2*margin) / spany

    dens = np.zeros((h, w), dtype=np.float32)

    def put_segment(x0,y0,x1,y1):
        #Sample along the segment
        dx, dy = x1 - x0, y1 - y0
        steps = max(2, int(max(abs(dx*scale_x), abs(dy*scale_y))))
        xs = np.linspace(x0, x1, steps)
        ys = np.linspace(y0, y1, steps)
        ix = np.clip(((xs - minx) * scale_x + margin).astype(np.int32), 0, w-1)
        iy = np.clip(((ys - miny) * scale_y + margin).astype(np.int32), 0, h-1)
        dens[iy, ix] += 1.0

    for path in paths:
        for (x0,y0),(x1,y1) in zip(path[:-1], path[1:]):
            put_segment(x0,y0,x1,y1)

    #Light blur for smoothness (cheap 3x3 box blur)
    k = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32)
    k /= k.sum()
    #Pad and convolve
    pad = 1
    padded = np.pad(dens, pad, mode='edge')
    out = np.zeros_like(dens)
    for i in range(h):
        for j in range(w):
            block = padded[i:i+3, j:j+3]
            out[i,j] = np.sum(block * k)
    return out

# --- UI --------------------------------------------------------------------
@dataclass
class Settings:
    start_count: int = 2000
    start_max: int = 1_000_000
    randomize: bool = True
    turn_deg: float = 8.6
    step0: float = 4.5
    decay: float = 0.985
    cmap: str = DEFAULT_CMAP
    gamma: float = 1.35
    animate: bool = False
    seed: Optional[int] = None

class CollatzTreeTk:
    def __init__(self, w=1400, h=850):
        self.root = tk.Tk()
        self.root.title("Collatz Conjecture — Tree Generator (Tk)")

        # --- Toolbar
        top = ttk.Frame(self.root, padding=(6,4,6,4))
        top.pack(side=tk.TOP, fill=tk.X)

        self.settings = Settings()

        #Start count
        ttk.Label(top, text="Paths:").pack(side=tk.LEFT, padx=(0,4))
        self.paths_var = tk.IntVar(value=self.settings.start_count)
        paths_scale = ttk.Scale(top, from_=100, to=10000, variable=self.paths_var, command=lambda _=None: self._set_status(), length=160)
        paths_scale.pack(side=tk.LEFT)

        #Randomize toggle
        self.rand_var = tk.BooleanVar(value=self.settings.randomize)
        ttk.Checkbutton(top, text="Random starts", variable=self.rand_var).pack(side=tk.LEFT, padx=(8,4))

        #Turn degrees slider
        ttk.Label(top, text="Turn°").pack(side=tk.LEFT, padx=(8,4))
        self.turn_var = tk.DoubleVar(value=self.settings.turn_deg)
        ttk.Scale(top, from_=2.0, to=30.0, variable=self.turn_var, command=lambda _=None: self._set_status(), length=120).pack(side=tk.LEFT)

        #Decay slider
        ttk.Label(top, text="Decay").pack(side=tk.LEFT, padx=(8,4))
        self.decay_var = tk.DoubleVar(value=self.settings.decay)
        ttk.Scale(top, from_=0.92, to=0.995, variable=self.decay_var, command=lambda _=None: self._set_status(), length=120).pack(side=tk.LEFT)

        #Step slider
        ttk.Label(top, text="Step").pack(side=tk.LEFT, padx=(8,4))
        self.step_var = tk.DoubleVar(value=self.settings.step0)
        ttk.Scale(top, from_=2.0, to=8.0, variable=self.step_var, command=lambda _=None: self._set_status(), length=120).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        #Colormap + gamma
        ttk.Label(top, text="Colormap:").pack(side=tk.LEFT, padx=(0,4))
        self.cmap_var = tk.StringVar(value=self.settings.cmap)
        ttk.OptionMenu(top, self.cmap_var, self.settings.cmap, *CMAPS).pack(side=tk.LEFT)

        ttk.Label(top, text="Gamma").pack(side=tk.LEFT, padx=(8,4))
        self.gamma_var = tk.DoubleVar(value=self.settings.gamma)
        ttk.Scale(top, from_=0.6, to=2.2, variable=self.gamma_var, command=lambda _=None: self._set_status(), length=120).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        #Render / Animate / Save
        ttk.Button(top, text="Render", command=self.render).pack(side=tk.LEFT, padx=(4,2))
        self.anim_var = tk.BooleanVar(value=self.settings.animate)
        ttk.Checkbutton(top, text="Animate", variable=self.anim_var).pack(side=tk.LEFT, padx=(8,2))
        ttk.Button(top, text="Save PNG", command=self.save_png).pack(side=tk.LEFT, padx=(8,0))

        # --- Canvas + status
        self.canvas = tk.Canvas(self.root, width=w, height=h, highlightthickness=0, bg="#ffffff")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status = ttk.Label(self.root, anchor="w", font=("Consolas", 10))
        self.status.pack(fill=tk.X)

        #State
        self.w, self.h = w, h
        self._imgtk: Optional[ImageTk.PhotoImage] = None
        self._current_img: Optional[Image.Image] = None
        self.root.bind("<Configure>", self._on_resize)

        #Initial draw
        self.render()

    # --- Helpers
    def _set_status(self, msg: Optional[str] = None):
        if msg is None:
            msg = (f"Paths={self.paths_var.get()}  Turn°={self.turn_var.get():.2f}  "
                   f"Decay={self.decay_var.get():.4f}  Step={self.step_var.get():.2f}  "
                   f"Gamma={self.gamma_var.get():.2f}  Cmap={self.cmap_var.get()}  "
                   f"Random={'ON' if self.rand_var.get() else 'OFF'}  "
                   f"Animate={'ON' if self.anim_var.get() else 'OFF'}  Size={self.w}x{self.h}")
        self.status.config(text=msg)

    def _on_resize(self, event):
        if event.widget is self.root:
            self.w = max(200, self.canvas.winfo_width())
            self.h = max(200, self.canvas.winfo_height())

    # --- Generation
    def _generate_start_numbers(self, count: int, max_n: int = 1_000_000, randomize: bool = True, seed: Optional[int]=None) -> List[int]:
        if seed is not None:
            random.seed(seed)
        if randomize:
            return [random.randint(2, max_n) for _ in range(count)]
        else:
            #Deterministic spread up to max_n
            step = max(1, max_n // count)
            return list(range(2, 2 + step*count, step))

    def _compute_paths(self) -> List[List[Tuple[float,float]]]:
        count = int(self.paths_var.get())
        turn = float(self.turn_var.get())
        step0 = float(self.step_var.get())
        decay = float(self.decay_var.get())

        starts = self._generate_start_numbers(count, randomize=self.rand_var.get())
        #Create a gentle canopy look by nudging the initial angle upward
        base_angle = math.radians(6.0)

        paths: List[List[Tuple[float,float]]] = []
        for s in starts:
            seq = collatz_sequence(s, max_len=4096)
            #Build path
            pts = sequence_to_path(seq, turn_deg=turn, step0=step0, decay=decay)
            #Rotate slightly
            rot = base_angle
            cr, sr = math.cos(rot), math.sin(rot)
            rpts = [(p[0]*cr - p[1]*sr, p[0]*sr + p[1]*cr) for p in pts]
            paths.append(rpts)
        return paths

    # --- Rendering
    def render(self):
        t0 = time.time()
        if self.anim_var.get():
            self._animate_render()
            return

        paths = self._compute_paths()
        dens = draw_paths_to_density(paths, self.w, self.h, margin=28)
        #Robust normalization
        lo = float(np.percentile(dens[dens>0], 2.0)) if np.any(dens>0) else 0.0
        hi = float(np.percentile(dens, 99.5)) if np.any(dens>0) else 1.0
        rgb = map_to_rgb(dens, self.cmap_var.get(), lo, hi, gamma=self.gamma_var.get(), smoothstep=True)
        img = Image.fromarray(rgb, mode="RGB")
        self._current_img = img
        self._imgtk = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._imgtk, anchor="nw")
        self._set_status(f"Rendered {len(paths)} paths in {time.time()-t0:.2f}s")

    def _animate_render(self):
        #Draw paths incrementally on a PIL image and update canvas via .after()
        self.canvas.delete("all")
        img = Image.new("RGB", (self.w, self.h), (255,255,255))
        draw = ImageDraw.Draw(img)

        paths = self._compute_paths()
        #Fit transform parameters by peeking at density mapping bbox
        #Reuse logic from draw_paths_to_density for consistent layout
        xs = np.concatenate([np.array([p[0] for p in path], dtype=np.float32) for path in paths])
        ys = np.concatenate([np.array([p[1] for p in path], dtype=np.float32) for path in paths])
        minx, maxx = float(xs.min()), float(xs.max())
        miny, maxy = float(ys.min()), float(ys.max())
        spanx = max(1e-6, maxx - minx)
        spany = max(1e-6, maxy - miny)
        target_aspect = self.h / self.w
        box_aspect = spany / spanx
        margin = 28
        if box_aspect > target_aspect:
            spanx = spany / target_aspect
            midx = (minx + maxx) * 0.5
            minx = midx - spanx/2
            maxx = midx + spanx/2
        else:
            spany = spanx * target_aspect
            midy = (miny + maxy) * 0.5
            miny = midy - spany/2
            maxy = midy + spany/2
        scale_x = (self.w - 2*margin) / spanx
        scale_y = (self.h - 2*margin) / spany

        def to_px(p):
            x = int((p[0]-minx)*scale_x + margin)
            y = int((p[1]-miny)*scale_y + margin)
            return x,y

        #Animated polyline drawing (batch steps for speed)
        batch = max(1, len(paths)//60)  # ~60 frames
        self._anim_state = {"idx": 0, "paths": paths, "img": img, "draw": draw, "to_px": to_px, "batch": batch}
        self._anim_tick()

    def _anim_tick(self):
        st = self._anim_state
        paths = st["paths"]
        draw = st["draw"]
        to_px = st["to_px"]
        b = st["batch"]
        end = min(len(paths), st["idx"]+b)

        #Draw batch
        for path in paths[st["idx"]:end]:
            if len(path) < 2: 
                continue
            #Thinner lines for animation + final colorize is via density in static mode,
            #But here monotone with alpha-like effect by using light color.
            for a,bp in zip(path[:-1], path[1:]):
                x0,y0 = to_px(a); x1,y1 = to_px(bp)
                draw.line((x0,y0,x1,y1), fill=(20,20,20), width=1)

        st["idx"] = end
        self._current_img = st["img"]
        self._imgtk = ImageTk.PhotoImage(st["img"])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._imgtk, anchor="nw")
        self._set_status(f"Animating... {end}/{len(paths)} paths")
        if end < len(paths):
            self.root.after(16, self._anim_tick)  #~60 FPS
        else:
            self._set_status(f"Animation complete: {len(paths)} paths")

    # --- Saving
    def save_png(self):
        if self._current_img is None:
            messagebox.showinfo("Info", "Nothing to save. Click Render first.")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        default = f"collatz_{ts}.png"
        path = filedialog.asksaveasfilename(
            title="Save PNG",
            defaultextension=".png",
            filetypes=[("PNG image","*.png")],
            initialfile=default,
        )
        if not path:
            return
        try:
            #Re-render at 2x for crisper output in static mode
            if not self.anim_var.get():
                w2, h2 = self.w*2, self.h*2
                paths = self._compute_paths()
                dens = draw_paths_to_density(paths, w2, h2, margin=56)
                lo = float(np.percentile(dens[dens>0], 2.0)) if np.any(dens>0) else 0.0
                hi = float(np.percentile(dens, 99.5)) if np.any(dens>0) else 1.0
                rgb = map_to_rgb(dens, self.cmap_var.get(), lo, hi, gamma=self.gamma_var.get(), smoothstep=True)
                Image.fromarray(rgb, mode="RGB").save(path, optimize=True)
            else:
                #For animation mode, just save current frame
                self._current_img.save(path, optimize=True)
            messagebox.showinfo("Saved", f"Saved: {os.path.abspath(path)}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    CollatzTreeTk().run()