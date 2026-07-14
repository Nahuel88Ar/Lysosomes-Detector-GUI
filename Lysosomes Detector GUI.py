#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Lysosomes Detector GUI
#Interactive Python software for automated 3D lysosome detection, cell segmentation, quantitative fluorescence analysis, and visualization from multichannel microscopy datasets.

"""
Sections:
1. Imports
2. GUI
3. Metadata parsing
4. Image loading
5. Lysosome detection
6. Cell segmentation
7. Quantification
8. Visualization
9. Napari editing
10. Export
"""

import os
import re
import numpy as np
import pandas as pd
import cv2
import imageio
import xml.etree.ElementTree as ET
from datetime import datetime

import czifile
from aicsimageio import AICSImage
import tifffile as tiff

try:
    import czifile
    _CZIFILE_IMPORT_ERROR = None
except Exception as e:
    czifile = None
    _CZIFILE_IMPORT_ERROR = e


from skimage.feature import blob_log
from skimage.filters import gaussian, threshold_local
from skimage.morphology import (
    remove_small_objects, binary_opening, binary_closing, ball, binary_erosion,binary_dilation
)
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes
from scipy.optimize import least_squares
from scipy.ndimage import binary_closing
import napari

from scipy.ndimage import gaussian_filter1d
import colorsys

# OPTION A helper imports (for marker peaks)
from skimage.morphology import h_maxima
from skimage.feature import peak_local_max

# ===============================
# GUI (single unified interface)
# ===============================
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

def get_user_config_gui(
    # basic defaults
    default_vxy_um=0.04,
    default_vz_um=None,          # if None and Z missing -> fallback to XY
    default_erode_mult=1.0,
    default_blob_threshold=0.001,

    # diameter filter defaults (optional; blank = no filter)
    default_diam_min_um=None,
    default_diam_max_um=None,

    # advanced defaults (ONLY the 3 you want)
    default_margin_um=1.2,          # µm
    default_overlap_alpha=0.6,      # unitless (0..1)
    default_neighbor_max_vox=12,    # voxels (kept in script only; NOT in GUI)
    default_viz_min_voxels=10000,   # voxels
    default_dist_smooth_sigma=4.0,
    default_h_maxima=4.0,
    
    # keep the rest as fixed defaults (not shown in GUI)
    default_max_reasonable_vxy_um=0.5,
    default_ch1_smooth_sigma=1.0,
    default_blob_min_sigma=0.8,  #0.8#0.6#last 1.5
    default_blob_max_sigma=2.5,  #3.0#2.5#last 4.0
    default_blob_num_sigma=12,    #10#15#last 6
    default_radial_max_radius_nm=400.0,#250.0 L1,L2,L4
    default_radial_dr_nm=5.0,    #10
    default_radial_min_drop_fraction=0.3,#last 0.5
    default_ch2_smooth_sigma=1.5,
    default_thresh_block_size=521,
    default_thresh_offset_std_mult=0.55,
    default_video_fps=8,

    default_launch_viewer=True,
    default_generate_videos=True,
):
    """
    One GUI to collect everything (NO presets JSON).
    Advanced section ONLY includes:
      - MARGIN_UM (µm)
      - OVERLAP_ALPHA (0..1)
      - VIZ_MIN_VOXELS (voxels)
    NOTE:
      - NEIGHBOR_MAX_VOX is kept as a fixed default in the script (NOT shown in GUI).
      - XY/Z overrides are NOT shown in GUI. If metadata is missing, the script asks later.
      - User can choose a lysosome DIAMETER interval to visualize + export interval datasets.
    """
    cfg = {"ok": False}

    root = tk.Tk()
    root.title("Lysosome + Cell Segmentation (GUI)")
    root.resizable(False, False)

    # Vars
    file_var = tk.StringVar(value="")
    out_var = tk.StringVar(value="")

    erode_var = tk.StringVar(value=str(default_erode_mult))
    blob_var  = tk.StringVar(value=str(default_blob_threshold))

    # Diameter interval (µm) - optional
    diam_min_var = tk.StringVar(value="" if default_diam_min_um is None else str(default_diam_min_um))
    diam_max_var = tk.StringVar(value="" if default_diam_max_um is None else str(default_diam_max_um))

    # Advanced vars (ONLY 3) + toggle
    show_adv = tk.BooleanVar(value=False)
    margin_var   = tk.StringVar(value=str(default_margin_um))
    overlap_var  = tk.StringVar(value=str(default_overlap_alpha))
    vizmin_var   = tk.StringVar(value=str(default_viz_min_voxels))
    distsmooth_var = tk.StringVar(value=str(default_dist_smooth_sigma))
    hmax_var       = tk.StringVar(value=str(default_h_maxima))

    fps_var       = tk.StringVar(value=str(default_video_fps))
    launch_viewer_var = tk.BooleanVar(value=bool(default_launch_viewer))
    gen_videos_var    = tk.BooleanVar(value=bool(default_generate_videos))

    def _suggest_output_dir(fp):
        if not fp:
            return ""
        raw_dir = os.path.dirname(fp)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(raw_dir, f"outputs_{stamp}")

    def browse_file():
        fp = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[("Image files", "*.tif *.tiff *.czi"), ("All files", "*.*")],
        )
        if fp:
            file_var.set(fp)
            if not out_var.get().strip():
                out_var.set(_suggest_output_dir(fp))

    def browse_output_dir():
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            fp = file_var.get().strip()
            if fp:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_var.set(os.path.join(d, f"outputs_{stamp}"))
            else:
                out_var.set(d)

    def _err(msg):
        messagebox.showerror("Invalid input", msg)
        raise ValueError(msg)

    def _float_required(s, name):
        s = (s or "").strip()
        if s == "":
            _err(f"{name} is required.")
        try:
            return float(s.replace(",", "."))
        except Exception:
            _err(f"{name} must be a number (got: {s})")

    def _float_optional(s, name):
        s = (s or "").strip()
        if s == "":
            return None
        try:
            return float(s.replace(",", "."))
        except Exception:
            _err(f"{name} must be a number (got: {s})")

    def _int_required(s, name):
        s = (s or "").strip()
        if s == "":
            _err(f"{name} is required.")
        try:
            return int(float(s.replace(",", ".")))
        except Exception:
            _err(f"{name} must be an integer (got: {s})")

    def _toggle_adv():
        if show_adv.get():
            adv_frame.grid()
        else:
            adv_frame.grid_remove()

    def run_clicked():
        fp = file_var.get().strip()
        if not fp:
            _err("Please select a file.")
        if not os.path.isfile(fp):
            _err("Selected file does not exist.")

        outd = out_var.get().strip() or _suggest_output_dir(fp)

        # Basic
        erode = _float_required(erode_var.get(), "ERODE_MULT")
        blobt = _float_required(blob_var.get(), "blob_log threshold")
        fps   = _int_required(fps_var.get(), "video FPS")

        # Diameter interval (optional)
        dmin = _float_optional(diam_min_var.get(), "Min lysosome diameter (µm)")
        dmax = _float_optional(diam_max_var.get(), "Max lysosome diameter (µm)")
        if (dmin is not None) and (dmin < 0):
            _err("Min lysosome diameter must be >= 0.")
        if (dmax is not None) and (dmax < 0):
            _err("Max lysosome diameter must be >= 0.")
        if (dmin is not None) and (dmax is not None) and (dmin > dmax):
            _err("Min lysosome diameter cannot be larger than Max lysosome diameter.")

        # Advanced (ONLY 3)
        margin  = _float_required(margin_var.get(), "MARGIN_UM (µm)")
        overlap = _float_required(overlap_var.get(), "OVERLAP_ALPHA (0..1)")
        vizmin  = _int_required(vizmin_var.get(), "VIZ_MIN_VOXELS (voxels)")
        dist_smooth_sigma = _float_required(distsmooth_var.get(),"DIST_SMOOTH_SIGMA")
        h_maxima = _float_required(hmax_var.get(),"H_MAXIMA")

        if not (0.0 <= overlap <= 1.0):
            _err("OVERLAP_ALPHA must be between 0 and 1.")

        cfg.update({
            "ok": True,
            "file_path": fp,
            "output_dir": outd,

            "ERODE_MULT": float(erode),
            "BLOB_THRESHOLD": float(blobt),

            # Diameter filter
            "DIAMETER_MIN_UM": None if dmin is None else float(dmin),
            "DIAMETER_MAX_UM": None if dmax is None else float(dmax),

            "DEFAULT_VX_VY_UM": float(default_vxy_um),
            "DEFAULT_VZ_UM": None if default_vz_um is None else float(default_vz_um),

            # keep sanity limit internal (not shown)
            "MAX_REASONABLE_VXY_UM": float(default_max_reasonable_vxy_um),

            # Advanced only (3)
            "MARGIN_UM": float(margin),
            "OVERLAP_ALPHA": float(overlap),
            "VIZ_MIN_VOXELS": int(vizmin),
            "DIST_SMOOTH_SIGMA": float(dist_smooth_sigma),
            "H_MAXIMA": float(h_maxima),

            # Kept default only (NOT in GUI)
            "NEIGHBOR_MAX_VOX": int(default_neighbor_max_vox),

            # fixed defaults (not in GUI)
            "CH1_SMOOTH_SIGMA": float(default_ch1_smooth_sigma),
            "BLOB_MIN_SIGMA": float(default_blob_min_sigma),
            "BLOB_MAX_SIGMA": float(default_blob_max_sigma),
            "BLOB_NUM_SIGMA": int(default_blob_num_sigma),

            "RADIAL_MAX_RADIUS_NM": float(default_radial_max_radius_nm),
            "RADIAL_DR_NM": float(default_radial_dr_nm),
            "RADIAL_MIN_DROP_FRACTION": float(default_radial_min_drop_fraction),

            "CH2_SMOOTH_SIGMA": float(default_ch2_smooth_sigma),
            "THRESH_BLOCK_SIZE": int(default_thresh_block_size),
            "THRESH_OFFSET_STD_MULT": float(default_thresh_offset_std_mult),

            "VIDEO_FPS": int(fps),
            "LAUNCH_VIEWER": bool(launch_viewer_var.get()),
            "GENERATE_VIDEOS": bool(gen_videos_var.get()),
        })

        root.destroy()

    def cancel_clicked():
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", cancel_clicked)

    pad = {"padx": 10, "pady": 6}
    frm = ttk.Frame(root)
    frm.grid(row=0, column=0, sticky="nsew", **pad)

    r = 0
    ttk.Label(frm, text="Image file:").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm, textvariable=file_var, width=60).grid(row=r, column=1, sticky="we")
    ttk.Button(frm, text="Browse...", command=browse_file).grid(row=r, column=2, sticky="e")
    r += 1

    ttk.Label(frm, text="Output folder:").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm, textvariable=out_var, width=60).grid(row=r, column=1, sticky="we")
    ttk.Button(frm, text="Browse...", command=browse_output_dir).grid(row=r, column=2, sticky="e")
    r += 1

    ttk.Separator(frm).grid(row=r, column=0, columnspan=3, sticky="we", pady=8)
    r += 1

    ttk.Label(frm, text="ERODE_MULT:").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm, textvariable=erode_var, width=20).grid(row=r, column=1, sticky="w")
    ttk.Label(frm, text="(unitless)").grid(row=r, column=2, sticky="w")
    r += 1

    ttk.Label(frm, text="blob_log threshold:").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm, textvariable=blob_var, width=20).grid(row=r, column=1, sticky="w")
    ttk.Label(frm, text="(unitless)").grid(row=r, column=2, sticky="w")
    r += 1

    ttk.Separator(frm).grid(row=r, column=0, columnspan=3, sticky="we", pady=8)
    r += 1

    ttk.Label(frm, text="Min lysosome diameter (µm):").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm, textvariable=diam_min_var, width=20).grid(row=r, column=1, sticky="w")
    ttk.Label(frm, text="blank = no filter").grid(row=r, column=2, sticky="w")
    r += 1

    ttk.Label(frm, text="Max lysosome diameter (µm):").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm, textvariable=diam_max_var, width=20).grid(row=r, column=1, sticky="w")
    ttk.Label(frm, text="blank = no filter").grid(row=r, column=2, sticky="w")
    r += 1

    ttk.Separator(frm).grid(row=r, column=0, columnspan=3, sticky="we", pady=8)
    r += 1

    ttk.Checkbutton(frm, text="Launch Napari viewer", variable=launch_viewer_var)\
        .grid(row=r, column=0, columnspan=2, sticky="w")
    ttk.Checkbutton(frm, text="Generate videos", variable=gen_videos_var)\
        .grid(row=r, column=2, sticky="w")
    r += 1

    ttk.Label(frm, text="Video FPS:").grid(row=r, column=0, sticky="w")
    ttk.Entry(frm, textvariable=fps_var, width=20).grid(row=r, column=1, sticky="w")
    ttk.Label(frm, text="(frames/sec)").grid(row=r, column=2, sticky="w")
    r += 1

    ttk.Separator(frm).grid(row=r, column=0, columnspan=3, sticky="we", pady=8)
    r += 1

    ttk.Checkbutton(frm, text="Show advanced settings", variable=show_adv, command=_toggle_adv)\
        .grid(row=r, column=0, columnspan=3, sticky="w")
    r += 1

    adv_frame = ttk.LabelFrame(frm, text="Advanced")
    adv_frame.grid(row=r, column=0, columnspan=3, sticky="we", pady=6)
    adv_frame.grid_remove()

    rr = 0
    def add_row(lbl, var, hint):
        nonlocal rr
        ttk.Label(adv_frame, text=lbl).grid(row=rr, column=0, sticky="w", padx=8, pady=3)
        ttk.Entry(adv_frame, textvariable=var, width=18).grid(row=rr, column=1, sticky="w", padx=8, pady=3)
        ttk.Label(adv_frame, text=hint).grid(row=rr, column=2, sticky="w", padx=8, pady=3)
        rr += 1

    add_row("MARGIN_UM:", margin_var, "µm (soft band around neuron mask)")
    add_row("OVERLAP_ALPHA:", overlap_var, "unitless (0..1)")
    add_row("VIZ_MIN_VOXELS:", vizmin_var, "voxels (hide small cells)")
    add_row("DIST_SMOOTH_SIGMA:", distsmooth_var,"watershed distance-map smoothing")
    add_row("H_MAXIMA:", hmax_var,"watershed seed detection")

    r += 1
    ttk.Separator(frm).grid(row=r, column=0, columnspan=3, sticky="we", pady=8)
    r += 1
    btns = ttk.Frame(frm)
    btns.grid(row=r, column=0, columnspan=3, sticky="e")
    ttk.Button(btns, text="Cancel", command=cancel_clicked).grid(row=0, column=0, padx=6)
    ttk.Button(btns, text="Run", command=run_clicked).grid(row=0, column=1, padx=6)

    root.mainloop()

    if not cfg.get("ok"):
        raise SystemExit("Cancelled.")
    return cfg


# ===============================
# Metadata parsing
# ===============================
def _parse_ome_xml(xml_text):
    if not xml_text:
        return None, None, None

    def grab(attr):
        m = re.search(fr'PhysicalSize{attr}="([\d\.eE+-]+)"', xml_text)
        return float(m.group(1)) if m else None

    return grab("X"), grab("Y"), grab("Z")


def _parse_czi_scaling(czi_text):
    """
    Robust CZI scaling parser:
    - Works when values are nested (<Value>...</Value>) or attributes
    - Works with namespaces
    - Converts meters -> µm
    """
    if not czi_text:
        return None, None, None

    if isinstance(czi_text, (bytes, bytearray)):
        czi_text = czi_text.decode("utf-8", errors="ignore")

    czi_text = czi_text.replace("\x00", "")

    def _to_float(s):
        if s is None:
            return None
        try:
            return float(str(s).strip().replace(",", "."))
        except Exception:
            return None

    def _to_um(val, unit_hint=None):
        if val is None:
            return None
        if unit_hint:
            u = str(unit_hint).strip().lower()
            if u in ("m", "meter", "metre", "meters", "metres"):
                return val * 1e6
            if u in ("µm", "um", "micron", "microns", "micrometer", "micrometre"):
                return val
            if u in ("nm", "nanometer", "nanometre", "nanometers", "nanometres"):
                return val / 1000.0

        # heuristic for CZI: meters ~ 1e-8
        if val < 1e-3:
            return val * 1e6
        if val < 10:
            return val
        if val < 1e5:
            return val / 1000.0
        return None

    try:
        root = ET.fromstring(czi_text)
    except Exception:
        def _grab(axis):
            mm = re.search(
                rf'<Distance[^>]*Id="{axis}"[^>]*>.*?<Value>\s*([0-9eE\+\-\.]+)\s*</Value>',
                czi_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            return _to_float(mm.group(1)) if mm else None

        return _to_um(_grab("X")), _to_um(_grab("Y")), _to_um(_grab("Z"))

    sx = sy = sz = None
    for d in root.findall(".//{*}Distance"):
        axis = d.attrib.get("Id") or d.attrib.get("id") or d.attrib.get("Axis") or d.attrib.get("axis")
        if not axis:
            continue
        axis = axis.upper()
        unit = d.attrib.get("Unit") or d.attrib.get("unit")

        valf = _to_float(d.attrib.get("Value") or d.attrib.get("value"))

        if valf is None:
            v_el = d.find(".//{*}Value")
            if v_el is not None and v_el.text:
                valf = _to_float(v_el.text)

        if valf is None:
            for child in d.iter():
                if child is d:
                    continue
                if str(child.tag).lower().endswith("value"):
                    valf = _to_float(child.attrib.get("Value") or child.attrib.get("value")) or _to_float(child.text)
                    if valf is not None:
                        break

        val_um = _to_um(valf, unit_hint=unit)
        if val_um is None:
            continue
        if axis == "X":
            sx = val_um
        elif axis == "Y":
            sy = val_um
        elif axis == "Z":
            sz = val_um

    return sx, sy, sz


def load_any(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".tif", ".tiff"):
        with tiff.TiffFile(file_path) as tf:
            arr = tf.asarray()
            try:
                ome_xml = tf.ome_metadata
            except Exception:
                ome_xml = None
            vx_um = vy_um = vz_um = None
            if ome_xml:
                vx_um, vy_um, vz_um = _parse_ome_xml(ome_xml)

        img = np.squeeze(arr)
        if img.ndim == 4:
            if img.shape[0] == 2:
                ch1, ch2 = img[0], img[1]
            elif img.shape[1] == 2:
                ch1, ch2 = img[:, 0], img[:, 1]
            elif img.shape[-1] == 2:
                ch1, ch2 = img[..., 0], img[..., 1]
            else:
                raise RuntimeError("Unexpected TIFF shape for 2 channels")
        else:
            raise RuntimeError("Unexpected TIFF shape")

        return ch1, ch2, (vx_um, vy_um, vz_um), {"type": "tiff"}
    
    if ext == ".czi":
        with czifile.CziFile(file_path) as cf:
            arr = cf.asarray()
            try:
                czi_xml = cf.metadata()
            except Exception:
                czi_xml = None

        vx_um = vy_um = vz_um = None
        if czi_xml:
            vx_um, vy_um, vz_um = _parse_czi_scaling(czi_xml)

        img = np.squeeze(arr)
        if img.ndim == 4:
            if img.shape[0] == 2:
                ch1, ch2 = img[0], img[1]
            elif img.shape[1] == 2:
                ch1, ch2 = img[:, 0], img[:, 1]
            elif img.shape[-1] == 2:
                ch1, ch2 = img[..., 0], img[..., 1]
            else:
                raise RuntimeError("Unexpected CZI shape for 2 channels")
        else:
            raise RuntimeError("Unexpected CZI shape")

        return ch1, ch2, (vx_um, vy_um, vz_um), {"type": "czi"}

def refine_radii_via_3d_gaussian_fit(
    img3d,
    blobs,
    vx_um,
    vy_um,
    vz_um,
    win_um=0.5,  #1.2#last 1
):
    """
    Fit 3D Gaussian to local cube around each blob.
    Returns blobs with blobs[:,3] = sigma_xy converted to XY pixels.
    """
    if blobs is None or len(blobs) == 0:
        return blobs

    img = img3d.astype(np.float32)
    Z, Y, X = img.shape
    out = blobs.copy().astype(np.float32)

    # window size in voxels
    rz = int(np.ceil(win_um / vz_um))
    ry = int(np.ceil(win_um / vy_um))
    rx = int(np.ceil(win_um / vx_um))

    px_um_xy = float(np.sqrt(vx_um * vy_um))

    for i, (zc, yc, xc, _) in enumerate(out):
        z0 = int(round(zc))
        y0 = int(round(yc))
        x0 = int(round(xc))

        if not (0 <= z0 < Z and 0 <= y0 < Y and 0 <= x0 < X):
            continue

        z1, z2 = max(0, z0 - rz), min(Z, z0 + rz + 1)
        y1, y2 = max(0, y0 - ry), min(Y, y0 + ry + 1)
        x1, x2 = max(0, x0 - rx), min(X, x0 + rx + 1)

        patch = img[z1:z2, y1:y2, x1:x2]
        if patch.size < 50:
            continue

        zz, yy, xx = np.mgrid[z1:z2, y1:y2, x1:x2]

        dz = (zz - z0) * vz_um
        dy = (yy - y0) * vy_um
        dx = (xx - x0) * vx_um

        coords = np.stack([dz.ravel(), dy.ravel(), dx.ravel()], axis=1)
        intens = patch.ravel()

        # Initial guesses
        A0 = intens.max() - intens.min()
        B0 = intens.min()
        sx0 = 0.2
        sy0 = 0.2
        sz0 = 0.2

        p0 = np.array([A0, B0, sx0, sy0, sz0])

        def residuals(p):
            A, B, sx, sy, sz = p
            if sx <= 0 or sy <= 0 or sz <= 0:
                return np.ones_like(intens) * 1e6

            model = A * np.exp(
                -(coords[:, 0]**2) / (2 * sz**2)
                -(coords[:, 1]**2) / (2 * sy**2)
                -(coords[:, 2]**2) / (2 * sx**2)
            ) + B

            return model - intens

        try:
            res = least_squares(residuals, p0, method='trf')
            A, B, sx, sy, sz = res.x

            # XY sigma average
            sigma_xy_um = 0.5 * (abs(sx) + abs(sy))  #0.5#0.4#0.3#original
                
            # Convert to pixel-equivalent radius for compatibility
            out[i, 3] = sigma_xy_um / px_um_xy

        except Exception:
            continue

    return out


def refine_radii_via_radial_intensity(
    img3d,
    blobs,
    vx_um,
    vy_um,
    vz_um,
    max_radius_nm=400,#last 400.0#L3 250
    dr_nm=5.0,  #10.0
    min_drop_fraction=0.3,  #0.5#last 0.4
):
    if blobs is None or len(blobs) == 0:
        return blobs

    img = img3d.astype(np.float32)
    Z, Y, X = img.shape
    blobs_out = blobs.copy().astype(np.float32)

    max_r_um = max_radius_nm / 1000.0
    dr_um = dr_nm / 1000.0

    r_edges = np.arange(0.0, max_r_um + dr_um, dr_um)
    if r_edges.size < 2:
        return blobs_out
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    px_um_xy = float(np.sqrt(vx_um * vy_um))

    for i, (zc, yc, xc, r_px_init) in enumerate(blobs_out):
        z0 = int(round(zc))
        y0 = int(round(yc))
        x0 = int(round(xc))

        if not (0 <= z0 < Z and 0 <= y0 < Y and 0 <= x0 < X):
            continue

        rz = max(1, int(np.ceil(max_r_um / vz_um)))
        ry = max(1, int(np.ceil(max_r_um / vy_um)))
        rx = max(1, int(np.ceil(max_r_um / vx_um)))

        z1, z2 = max(0, z0 - rz), min(Z, z0 + rz + 1)
        y1, y2 = max(0, y0 - ry), min(Y, y0 + ry + 1)
        x1, x2 = max(0, x0 - rx), min(X, x0 + rx + 1)
        if z1 >= z2 or y1 >= y2 or x1 >= x2:
            continue

        patch = img[z1:z2, y1:y2, x1:x2]

        zz, yy, xx = np.mgrid[z1:z2, y1:y2, x1:x2]
        dz_um = (zz - z0) * vz_um
        dy_um = (yy - y0) * vy_um
        dx_um = (xx - x0) * vx_um
        r_um = np.sqrt(dz_um**2 + dy_um**2 + dx_um**2)

        mask = (r_um <= max_r_um)
        if not np.any(mask):
            continue

        r_vals = r_um[mask].ravel()
        I_vals = patch[mask].ravel()

        bin_idx = np.digitize(r_vals, r_edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < r_centers.size)
        if not np.any(valid):
            continue

        bin_idx = bin_idx[valid]
        I_vals = I_vals[valid]

        sums = np.bincount(bin_idx, weights=I_vals, minlength=r_centers.size)
        counts = np.bincount(bin_idx, minlength=r_centers.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            prof = sums / np.maximum(counts, 1)

        have = counts > 0
        if not np.any(have):
            continue

        r_prof = r_centers[have]
        I_prof = prof[have].astype(np.float32)

        # Smooth radial profile
        I_smooth = gaussian_filter1d(I_prof, sigma=1.0)

        I_max = float(I_smooth.max())
        I_min = float(I_smooth.min())

        if I_max <= 0:
            continue

        # Peak index
        peak_idx = int(np.argmax(I_smooth))
        n_bins = len(I_smooth)

        # --- Estimate left/right intensities near edges ---
        left_probe = peak_idx
        while left_probe > 0 and I_smooth[left_probe] >= I_min:
            left_probe -= 1

        right_probe = peak_idx
        while right_probe < n_bins - 1 and I_smooth[right_probe] >= I_min:
            right_probe += 1

        I_left = float(I_smooth[left_probe])
        I_right = float(I_smooth[right_probe])

        # --- Symmetry check ---
        symmetry_ratio = abs(I_left - I_right)/ max(I_max, 1e-9)

        # Choose baseline depending on symmetry
        if symmetry_ratio == 0:
            # symmetric → use global minimum
            baseline = I_min
        else:
            # asymmetric → use upper side
            baseline = max(I_left, I_right)

        # --- Compute half-maximum based on chosen baseline ---
        I_half = baseline + 0.5 * (I_max - baseline)

        # --- Find FWHM crossings ---
        left_idx = peak_idx
        while left_idx > 0 and I_smooth[left_idx] >= I_half:
            left_idx -= 1
        if left_idx < peak_idx and I_smooth[left_idx] < I_half:
            left_idx += 1
       
        right_idx = peak_idx
        while right_idx < n_bins - 1 and I_smooth[right_idx] >= I_half:
            right_idx += 1
        if right_idx > peak_idx and I_smooth[right_idx] < I_half:
            right_idx -= 1

        # --- Validate indices ---
        if right_idx <= left_idx:
            continue

        r_left = float(r_prof[left_idx])
        r_right = float(r_prof[right_idx])

        if np.isnan(r_left) or np.isnan(r_right):
            continue

        if r_right <= r_left:
            continue

        # --- Compute radius (FWHM / 2) ---
        radius_um = 0.5 * (r_right - r_left)

        if radius_um <= 0:
            continue

        # Convert to pixels
        r_fwhm_px = radius_um / max(px_um_xy, 1e-9)

        # Optional: choose behavior
        # Option A (true refinement):
        blobs_out[i, 3] = float(r_fwhm_px)

        # Option B (only grow, never shrink):
        # blobs_out[i, 3] = max(float(r_px_init), float(r_fwhm_px))
    return blobs_out

# ===============================
# AUTO MORPHOLOGY
# ===============================
from scipy.ndimage import distance_transform_edt as _edt
from skimage.measure import label as _label, regionprops as _regionprops


def _equiv_radius_from_area(px_area):
    return float(np.sqrt(max(px_area, 1.0) / np.pi))


def _component_size_percentile(mask_bool, pct=0.2):
    lab = _label(mask_bool)
    sizes = [r.area for r in _regionprops(lab) if r.label != 0]
    if not sizes:
        return 0.0
    sizes = np.array(sorted(sizes))
    return float(np.percentile(sizes, pct * 100.0))


def auto_morphology_params(
    neuron_mask,
    vx_um,
    vy_um,
    vz_um,
    p_open=2,
    p_close=1.8,
    p_erode=0.15,
    min_r_open=1,
    min_r_close=1,
    max_r=12,
):
    din = _edt(neuron_mask)
    r_in_med = float(np.median(din[neuron_mask])) if np.any(neuron_mask) else 0.0
    small_px = _component_size_percentile(neuron_mask, pct=0.20)
    r_small = _equiv_radius_from_area(small_px)

    r_close = int(np.clip(round(p_close * r_in_med), min_r_close, max_r))
    r_open = int(np.clip(round(p_open * r_small), min_r_open, max_r))
    r_erode = int(np.clip(round(p_erode * r_in_med), 0, max_r))
    return max(r_open, 0), max(r_close, 0), max(r_erode, 0)


def apply_morphology_auto(neuron_mask, vx_um, vy_um, vz_um,
                          ERODE_MULT=1.0, mode="dt", ch2_for_scoring=None,
                          area_stability=(0.85, 1.15)):
    r_open, r_close, r_erode = auto_morphology_params(neuron_mask, vx_um, vy_um, vz_um)

    max_r = 12
    # NOTE: keeps your behavior: ERODE_MULT is additive (voxels)
    r_erode = int(np.clip(int(round(r_erode + float(ERODE_MULT))), 0, max_r))

    def _refine(mask, ro, rc, re_):
        out = mask.copy()
        if ro > 0:
            out = binary_opening(out, ball(ro))
        if rc > 0:
            out = binary_closing(out, ball(rc))
        if re_ > 0:
            out = binary_erosion(out, ball(re_))
        out = remove_small_objects(out, min_size=max(8, int(np.sum(out) * 1e-5)), connectivity=3)
        return out

    refined = _refine(neuron_mask, r_open, r_close, r_erode)
    return refined, {'open': r_open, 'close': r_close, 'erode': r_erode}


# ===============================
# FULL-SIZE EXPORT HELPERS
# ===============================
def _norm_u8_stack(vol):
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        out = (np.clip((vol - vmin) / (vmax - vmin), 0, 1) * 255.0).astype(np.uint8)
    else:
        out = np.zeros_like(vol, dtype=np.uint8)
    return out


def make_label_colormap(n_labels, seed_hue=0.0):
    colors = np.zeros((n_labels + 1, 3), dtype=np.uint8)
    if n_labels <= 0:
        return colors

    for i in range(1, n_labels + 1):
        h = (seed_hue + (i - 1) / max(n_labels, 1)) % 1.0
        s = 1.0
        v = 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors[i] = (int(255 * r), int(255 * g), int(255 * b))
    return colors


def export_fullsize_overlay_stack(
    img_ch1,
    img_ch2_raw,
    cell_seg_viz,
    df,
    vx_um, vy_um, vz_um,
    output_dir,
    alpha_labels=0.45,
    draw_only_inside=True,
    fps=8,
    basename="FULLSIZE_overlay_CellID_Lysosomes",
):
    os.makedirs(output_dir, exist_ok=True)

    ch1_u8 = _norm_u8_stack(img_ch1.astype(np.float32))
    ch2_u8 = _norm_u8_stack(img_ch2_raw.astype(np.float32))

    Z, H, W = ch2_u8.shape
    n_labels = int(cell_seg_viz.max()) if isinstance(cell_seg_viz, np.ndarray) else 0
    cmap = make_label_colormap(n_labels, seed_hue=0.13)

    use_df = None
    if isinstance(df, pd.DataFrame) and len(df) > 0 and {"z_um", "y_um", "x_um", "radius_um"}.issubset(df.columns):
        if draw_only_inside and "location_ch2" in df.columns:
            use_df = df[df["location_ch2"] == "cell"].copy()
        else:
            use_df = df.copy()

        use_df = use_df[
            np.isfinite(use_df["z_um"]) &
            np.isfinite(use_df["y_um"]) &
            np.isfinite(use_df["x_um"]) &
            np.isfinite(use_df["radius_um"])
        ].copy()

    px_um_xy = float(np.sqrt(vx_um * vy_um))
    frames = np.zeros((Z, H, W, 3), dtype=np.uint8)

    for z in range(Z):
        base = np.dstack([ch1_u8[z], ch2_u8[z], ch1_u8[z]]).astype(np.float32)

        lab2d = cell_seg_viz[z].astype(np.int32)
        lab_rgb = cmap[lab2d]
        lab_rgb_f = lab_rgb.astype(np.float32)

        mask = (lab2d > 0)[..., None].astype(np.float32)
        out = base * (1.0 - alpha_labels * mask) + lab_rgb_f * (alpha_labels * mask)

        if use_df is not None and len(use_df) > 0:
            zc = (use_df["z_um"].to_numpy() / vz_um).astype(float)
            yc = (use_df["y_um"].to_numpy() / vy_um).astype(float)
            xc = (use_df["x_um"].to_numpy() / vx_um).astype(float)
            r_um = use_df["radius_um"].to_numpy().astype(float)

            dz_um = np.abs(zc - z) * vz_um
            hits = dz_um <= r_um
            if np.any(hits):
                r_proj_um = np.sqrt(np.clip(r_um[hits]**2 - dz_um[hits]**2, 0.0, None))
                r_proj_px = r_proj_um / max(px_um_xy, 1e-12)

                ys = np.rint(yc[hits]).astype(int)
                xs = np.rint(xc[hits]).astype(int)

                out_u8 = np.clip(out, 0, 255).astype(np.uint8)
                for y, x, rp in zip(ys, xs, r_proj_px):
                    rr = int(max(3, round(rp)))
                    if 0 <= y < H and 0 <= x < W and rr > 0:
                        cv2.circle(out_u8, (x, y), rr, (0, 0, 0), 4, lineType=cv2.LINE_AA)
                        cv2.circle(out_u8, (x, y), rr, (0, 255, 255), 2, lineType=cv2.LINE_AA)
                out = out_u8.astype(np.float32)

        frames[z] = np.clip(out, 0, 255).astype(np.uint8)

    tiff_path = os.path.join(output_dir, f"{basename}.tif")
    tiff.imwrite(tiff_path, frames, photometric="rgb")
    print("Saved full-size RGB TIFF stack:", tiff_path)

    mp4_path = os.path.join(output_dir, f"{basename}.mp4")
    try:
        with imageio.get_writer(
            mp4_path,
            fps=int(fps),
            format="FFMPEG",
            codec="libx264",
            macro_block_size=None
        ) as w:
            for fr in frames:
                w.append_data(fr)
        print("Saved full-size MP4:", mp4_path)
    except Exception as e:
        gif_path = os.path.join(output_dir, f"{basename}.gif")
        imageio.mimsave(gif_path, list(frames), fps=int(fps))
        print("FFMPEG failed, saved GIF instead:", gif_path, "Error:", e)

    return tiff_path, mp4_path


# ===============================
# Helper: ask for voxel sizes only if metadata missing
# ===============================
def _ask_missing_scale(param_label, default_val, unit_text):
    rr = tk.Tk()
    rr.withdraw()
    try:
        use_def = messagebox.askyesno(
            "Missing metadata",
            f"{param_label} is missing in metadata.\n\n"
            f"Use default: {default_val} {unit_text}?\n\n"
            f"Yes = use default\nNo = enter new value"
        )
        val = float(default_val)
        if not use_def:
            v = simpledialog.askfloat(
                "Enter value",
                f"Enter {param_label} ({unit_text}):",
                initialvalue=float(default_val),
                minvalue=0.0
            )
            if v is not None:
                val = float(v)
        return float(val)
    finally:
        rr.destroy()


# ===============================
# Napari EDIT block adapters (REQUIRED)
# ===============================

NEURITE_MODE = False
EDIT_LYSOSOME_TABLE_IN_NAPARI = True
LYSOSOME_EDITED_CSV = "lysosomes_with_cell_vs_outside_EDITED.csv"


def attach_all_blob_fields(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    for col in ("z_um", "y_um", "x_um"):
        if col not in df.columns:
            df[col] = np.nan

    if "radius_um" not in df.columns and "diameter_um" in df.columns:
        df["radius_um"] = pd.to_numeric(df["diameter_um"], errors="coerce") / 2.0
    if "diameter_um" not in df.columns and "radius_um" in df.columns:
        df["diameter_um"] = pd.to_numeric(df["radius_um"], errors="coerce") * 2.0

    if "location_ch2" not in df.columns:
        df["location_ch2"] = "outside"
    df["location_ch2"] = df["location_ch2"].astype(str)

    if "cell_id_ch2" not in df.columns:
        df["cell_id_ch2"] = 0
    df["cell_id_ch2"] = pd.to_numeric(df["cell_id_ch2"], errors="coerce").fillna(0).astype(int)

    if "peak_gray" not in df.columns:
        df["peak_gray"] = np.nan

    for col in ("cell_id_serial", "lys_id_serial"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


# ===============================
# MAIN
# ===============================
DEFAULT_VX_VY_UM = 0.04
DEFAULT_VZ_UM = None

cfg = get_user_config_gui(
    default_vxy_um=DEFAULT_VX_VY_UM,
    default_vz_um=DEFAULT_VZ_UM,
    default_erode_mult=1.0,
    default_blob_threshold=0.001,
)

file_path = cfg["file_path"]
output_dir = cfg["output_dir"]
os.makedirs(output_dir, exist_ok=True)

def outpath(name):
    return os.path.join(output_dir, name)

print("Selected file:", file_path)
print("Outputs will be saved to:", output_dir)

ERODE_MULT = cfg["ERODE_MULT"]
BLOB_THRESHOLD = cfg["BLOB_THRESHOLD"]

DIAMETER_MIN_UM = cfg["DIAMETER_MIN_UM"]
DIAMETER_MAX_UM = cfg["DIAMETER_MAX_UM"]

MAX_REASONABLE_VXY_UM = cfg["MAX_REASONABLE_VXY_UM"]
MARGIN_UM = cfg["MARGIN_UM"]
OVERLAP_ALPHA = cfg["OVERLAP_ALPHA"]
NEIGHBOR_MAX_VOX = cfg["NEIGHBOR_MAX_VOX"]
VIZ_MIN_VOXELS = cfg["VIZ_MIN_VOXELS"]
DIST_SMOOTH_SIGMA = cfg["DIST_SMOOTH_SIGMA"]
H_MAXIMA = cfg["H_MAXIMA"]

CH1_SMOOTH_SIGMA = cfg["CH1_SMOOTH_SIGMA"]
BLOB_MIN_SIGMA = cfg["BLOB_MIN_SIGMA"]
BLOB_MAX_SIGMA = cfg["BLOB_MAX_SIGMA"]
BLOB_NUM_SIGMA = cfg["BLOB_NUM_SIGMA"]

RADIAL_MAX_RADIUS_NM = cfg["RADIAL_MAX_RADIUS_NM"]
RADIAL_DR_NM = cfg["RADIAL_DR_NM"]
RADIAL_MIN_DROP_FRACTION = cfg["RADIAL_MIN_DROP_FRACTION"]

CH2_SMOOTH_SIGMA = cfg["CH2_SMOOTH_SIGMA"]
THRESH_BLOCK_SIZE = cfg["THRESH_BLOCK_SIZE"]
THRESH_OFFSET_STD_MULT = cfg["THRESH_OFFSET_STD_MULT"]

FPS = cfg["VIDEO_FPS"]
LAUNCH_VIEWER = cfg["LAUNCH_VIEWER"]
GENERATE_VIDEOS = cfg["GENERATE_VIDEOS"]

# Load data + metadata
img_ch1, img_ch2, (vx_um, vy_um, vz_um), meta = load_any(file_path)
print(f"[metadata] vx_um={vx_um}  vy_um={vy_um}  vz_um={vz_um} ")

if vx_um is None or vy_um is None:
    vxy = _ask_missing_scale("XY pixel size", cfg["DEFAULT_VX_VY_UM"], "µm/px")
    vx_um = vy_um = float(vxy)

if vz_um is None:
    default_z = cfg["DEFAULT_VZ_UM"] if cfg["DEFAULT_VZ_UM"] is not None else float(vx_um)
    vz_um = _ask_missing_scale("Z step size", default_z, "µm/slice")

if vx_um > MAX_REASONABLE_VXY_UM:
    raise ValueError(f"XY pixel size too large: {vx_um} µm/px")

px_um_xy = float(np.sqrt(vx_um * vy_um))
px_um = px_um_xy 
voxel_um3 = vx_um * vy_um * vz_um
print(f"Voxel size (µm): X={vx_um}, Y={vy_um}, Z={vz_um}")

# ===== Aliases =====
image = img_ch1#lysosomes
image_2 = img_ch2#cells

# ==========================================
# Lysosome detection (Ch1)
# ==========================================
image_smooth = gaussian(image, sigma=CH1_SMOOTH_SIGMA)

blobs = blob_log(
    image_smooth,
    min_sigma=BLOB_MIN_SIGMA,
    max_sigma=BLOB_MAX_SIGMA,
    num_sigma=BLOB_NUM_SIGMA,
    threshold=BLOB_THRESHOLD
)

if blobs is None:
    blobs = np.zeros((0, 4), dtype=float)

if len(blobs) > 0:
    blobs = blobs.astype(np.float32)  #ORIGINAL
    # Ignore LoG sigma radius
    blobs[:, 3] = 0
    
    blobs = refine_radii_via_3d_gaussian_fit(
        image_smooth,
        blobs,
        vx_um,
        vy_um,
        vz_um,
        win_um=0.5  #0.4#last 1
    )
    
    blobs = refine_radii_via_radial_intensity(
        image_smooth,
        blobs,
        vx_um, vy_um, vz_um,
        max_radius_nm=RADIAL_MAX_RADIUS_NM,
        dr_nm=RADIAL_DR_NM,
        min_drop_fraction=RADIAL_MIN_DROP_FRACTION
    )

# Peak intensity in raw 16-bit Ch1
peak_gray = np.zeros(len(blobs), dtype=np.uint16)
Z0, Y0, X0 = image.shape
rad = 1
for i, (zc, yc, xc, _) in enumerate(blobs):
    zc_i = int(round(zc))
    yc_i = int(round(yc))
    xc_i = int(round(xc))

    z1, z2 = max(0, zc_i - rad), min(Z0, zc_i + rad + 1)
    y1, y2 = max(0, yc_i - rad), min(Y0, yc_i + rad + 1)
    x1, x2 = max(0, xc_i - rad), min(X0, xc_i + rad + 1)

    peak_gray[i] = np.max(image[z1:z2, y1:y2, x1:x2]).astype(np.uint16)

if len(blobs) > 0:
    z_um = blobs[:, 0] * vz_um
    y_um = blobs[:, 1] * vy_um
    x_um = blobs[:, 2] * vx_um
    radius_um = blobs[:, 3] * px_um_xy#0.7 GOOD
 
    MAX_RADIUS_UM = 0.4

    # clamp radii
    radius_um = np.clip(radius_um, 0.0, MAX_RADIUS_UM)

    # write back into blobs so EVERYTHING downstream uses capped values
    blobs[:, 3] = radius_um / px_um_xy

    diameter_um = 2 * radius_um
    volume_um3 = (4 / 3) * np.pi * radius_um**3
    blob_ids = np.arange(1, len(blobs) + 1, dtype=int)
else:
    z_um = y_um = x_um = radius_um = diameter_um = volume_um3 = np.array([])
    blob_ids = np.array([], dtype=int)

df = pd.DataFrame({
    "id": blob_ids,
    "z_um": z_um,
    "y_um": y_um,
    "x_um": x_um,
    "radius_um": radius_um,
    "diameter_um": diameter_um,
    "volume_um3": volume_um3,
    "peak_gray": peak_gray,
})
df.to_csv(outpath("lysosome_blobs_regions.csv"), index=False)
print("Saved:", outpath("lysosome_blobs_regions.csv"))


def _unique_radii_within_5pct(radius_series, low=0.0, high=0.4, max_frac=0.05):  #0.4
    arr = radius_series.to_numpy().astype(float)
    uniq = np.empty_like(arr, dtype=float)
    used = set()
    for i, r in enumerate(arr):
        base = float(np.clip(r, low, high))
        dev = max(max_frac * max(abs(base), 1e-6), 1e-9)
        if base not in used:
            uniq[i] = base
            used.add(base)
            continue
        found = False
        step = dev / 20.0
        for k in range(1, 401):
            sgn = 1.0 if (k % 2 == 1) else -1.0
            cand = float(np.clip(base + sgn * min(dev, k * step), low, high))
            if abs(cand - base) <= dev and cand not in used:
                uniq[i] = cand
                used.add(cand)
                found = True
                break
        if not found:
            cand = float(np.clip(np.nextafter(base, high), low, high))
            uniq[i] = cand
            used.add(cand)
    return pd.Series(uniq, index=radius_series.index, name="radius_um")


if len(df) > 0:
    df_unique = df.copy()
    df_unique["radius_um"] = _unique_radii_within_5pct(df_unique["radius_um"], low=0.0, high=0.4, max_frac=0.05)  #0.4
    df_unique["diameter_um"] = 2.0 * df_unique["radius_um"]
    df_unique["volume_um3"]  = (4.0 / 3.0) * np.pi * (df_unique["radius_um"] ** 3)
    df_unique.to_csv(outpath("lysosome_blobs_regions_unique_radius.csv"), index=False)
    print("Saved:", outpath("lysosome_blobs_regions_unique_radius.csv"))
    df = df_unique.copy()

df_all = df.copy()

# ==========================================
# Segmentation (CELL vs OUTSIDE)
# ==========================================
vol = image_2.astype(np.float32)
#vol = image.astype(np.float32)
vmin, vmax = float(vol.min()), float(vol.max())
if vmax > vmin:
    vol = (vol - vmin) / (vmax - vmin)
else:
    vol[:] = 0.0

ch2 = gaussian(vol, sigma=CH2_SMOOTH_SIGMA, preserve_range=True)

neuron_mask = np.zeros_like(ch2, dtype=bool)
for z in range(ch2.shape[0]):
    R = ch2[z]
    t = threshold_local(R, block_size=THRESH_BLOCK_SIZE, offset=-THRESH_OFFSET_STD_MULT * np.std(R))
    neuron_mask[z] = R > t

neuron_mask_auto, chosen = apply_morphology_auto(neuron_mask, vx_um, vy_um, vz_um, ERODE_MULT=ERODE_MULT, mode="dt")
print(f"[auto-morphology] radii -> open:{chosen['open']}  close:{chosen['close']}  erode:{chosen['erode']}")

total_mask = neuron_mask_auto.copy()

neuron_mask_for_segmentation = total_mask
neuron_mask_for_quantification = total_mask

# ======================================================================
# Markers from distance peaks (h-maxima / peak_local_max)
# ======================================================================
dist_inside = edt(neuron_mask_for_segmentation).astype(np.float32)

dist_smooth = gaussian(
    dist_inside,
    sigma=DIST_SMOOTH_SIGMA,
    preserve_range=True
).astype(np.float32)

maxima = h_maxima(dist_smooth, h=H_MAXIMA)

markers = label(maxima, connectivity=3)
n_markers = int(markers.max())
print(f"[markers] seeds found: {n_markers}")

if n_markers > 0:
    cell_seg = watershed(
        -dist_smooth,
        markers=markers,
        mask=neuron_mask_for_segmentation
    )
else:
    cell_seg = np.zeros_like(neuron_mask_for_segmentation, dtype=np.int32)

print(f"Detected {int(cell_seg.max())} cells.")
print("filled cell voxels:", int(neuron_mask_for_segmentation.sum()))
print("5-voxel membrane voxels:", int(neuron_mask_for_quantification.sum()))

# Debug exports
tiff.imwrite(outpath("debug_filled_cell_mask.tif"), neuron_mask_for_segmentation.astype(np.uint8) * 255)
tiff.imwrite(outpath("debug_5vox_membrane_shell.tif"), neuron_mask_for_quantification.astype(np.uint8) * 255)
tiff.imwrite(outpath("debug_dist_inside.tif"), dist_inside.astype(np.float32))
tiff.imwrite(outpath("debug_markers.tif"), markers.astype(np.uint16))
tiff.imwrite(outpath("debug_cell_seg.tif"), cell_seg.astype(np.uint16))

print("Saved debug filled-cell, membrane-shell, markers, and cell segmentation stacks.")

# ==========================================
# Visualization-only filtering (hide tiny cells) + serial IDs
# ==========================================
cell_seg_viz = cell_seg.copy()
cell_id_map_viz = {}  # original_id -> serial_id

if isinstance(cell_seg_viz, np.ndarray) and cell_seg_viz.max() > 0:
    counts_viz = np.bincount(cell_seg_viz.ravel().astype(np.int64))
    tiny_labels = np.where(counts_viz < VIZ_MIN_VOXELS)[0]
    tiny_labels = tiny_labels[tiny_labels > 0]
    if tiny_labels.size > 0:
        cell_seg_viz[np.isin(cell_seg_viz, tiny_labels)] = 0

    unique_labels = np.unique(cell_seg_viz)
    unique_labels = unique_labels[unique_labels > 0]

    if unique_labels.size > 0:
        new_seg = np.zeros_like(cell_seg_viz, dtype=np.int32)
        for new_id, old_id in enumerate(unique_labels, start=1):
            new_seg[cell_seg_viz == old_id] = new_id
            cell_id_map_viz[int(old_id)] = int(new_id)
        cell_seg_viz = new_seg

cell_mask_viz = (cell_seg_viz > 0)

# ==========================================
# Distance/overlap-aware classification helpers
# ==========================================
dist_out_um = _edt(~neuron_mask, sampling=(vz_um, vy_um, vx_um)).astype(np.float32)
soft_cell_mask = neuron_mask | (dist_out_um <= MARGIN_UM)


def nearest_cell_label(z, y, x, max_r=NEIGHBOR_MAX_VOX):
    Z, Y, X = cell_seg.shape
    for r in range(1, max_r + 1):
        z1, z2 = max(0, z - r), min(Z, z + r + 1)
        y1, y2 = max(0, y - r), min(Y, y + r + 1)
        x1, x2 = max(0, x - r), min(X, x + r + 1)
        patch = cell_seg[z1:z2, y1:y2, x1:x2]
        lab = patch[patch > 0]
        if lab.size:
            return int(np.bincount(lab.ravel()).argmax())
    return 0


def sphere_overlap_fraction(zc_um, yc_um, xc_um, r_um, mask_bool):
    if r_um <= 0:
        return 0.0
    zc = int(round(zc_um / vz_um))
    yc = int(round(yc_um / vy_um))
    xc = int(round(xc_um / vx_um))

    rz = max(1, int(np.ceil(r_um / vz_um)))
    ry = max(1, int(np.ceil(r_um / vy_um)))
    rx = max(1, int(np.ceil(r_um / vx_um)))

    Z, Y, X = mask_bool.shape
    z1, z2 = max(0, zc - rz), min(Z, zc + rz + 1)
    y1, y2 = max(0, yc - ry), min(Y, yc + ry + 1)
    x1, x2 = max(0, xc - rx), min(X, xc + rx + 1)
    if z1 >= z2 or y1 >= y2 or x1 >= x2:
        return 0.0

    zz, yy, xx = np.mgrid[z1:z2, y1:y2, x1:x2]
    dz = (zz - zc) * vz_um
    dy = (yy - yc) * vy_um
    dx = (xx - xc) * vx_um
    sphere = (dz * dz + dy * dy + dx * dx) <= (r_um * r_um)

    if not np.any(sphere):
        return 0.0

    in_mask = mask_bool[z1:z2, y1:y2, x1:x2] & sphere
    return float(in_mask.sum()) / float(sphere.sum())

# ==========================================
# Map lysosomes to (cell/outside) with per-cell IDs  (FULL dataset)
# ==========================================
location_ch2 = []
cell_id_list = []
df = df_all.copy()

if len(df) > 0:
    Z, Y, X = neuron_mask.shape
    for (zc_um, yc_um, xc_um, r_um) in df[["z_um", "y_um", "x_um", "radius_um"]].to_numpy():
        zz = int(round(zc_um / vz_um))
        yy = int(round(yc_um / vy_um))
        xx = int(round(xc_um / vx_um))

        inside_hard = (0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X and neuron_mask[zz, yy, xx])
        inside_soft = (0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X and soft_cell_mask[zz, yy, xx])

        is_inside = bool(inside_hard)
        if not is_inside and inside_soft:
            is_inside = True
        if not is_inside:
            frac = sphere_overlap_fraction(zc_um, yc_um, xc_um, r_um, neuron_mask)
            if frac >= OVERLAP_ALPHA:
                is_inside = True

        if is_inside:
            cid = 0
            if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                cid = int(cell_seg[zz, yy, xx]) if cell_seg[zz, yy, xx] != 0 else nearest_cell_label(zz, yy, xx)
            location_ch2.append("cell")
            cell_id_list.append(cid)
        else:
            location_ch2.append("outside")
            cell_id_list.append(0)

    df["location_ch2"] = location_ch2
    df["cell_id_ch2"]  = cell_id_list
    df["cell_id_ch2_viz"] = df["cell_id_ch2"].map(cell_id_map_viz).fillna(0).astype(int)

    df.groupby("location_ch2").size().reset_index(name="count").to_csv(
        outpath("lysosome_counts_cell_vs_outside.csv"), index=False
    )

    (df[df["location_ch2"] == "cell"]
        .groupby("cell_id_ch2").size()
        .reset_index(name="count")
        .to_csv(outpath("lysosome_counts_by_cell.csv"), index=False))

    df["lys_id_in_cell"] = 0
    mask_in = (df["location_ch2"] == "cell") & (df["cell_id_ch2"] > 0)
    df_sorted = df.loc[mask_in].sort_values(["cell_id_ch2", "z_um", "y_um", "x_um"]).copy()
    df.loc[df_sorted.index, "lys_id_in_cell"] = (df_sorted.groupby("cell_id_ch2").cumcount().to_numpy() + 1).astype(int)

    df.to_csv(outpath("lysosomes_with_cell_vs_outside.csv"), index=False)

    lys_serial_counts = (
        df[df["lys_id_in_cell"] > 0]
        .groupby("cell_id_ch2")["lys_id_in_cell"]
        .max()
        .reset_index()
        .rename(columns={"lys_id_in_cell": "lysosomes_in_cell"})
    )

    # --- NEW: compute cell volume (µm³) for each serial cell ID ---
    counts_serial = np.bincount(cell_seg_viz.ravel().astype(np.int64))
    vol_serial_um3 = counts_serial.astype(np.float64) * float(voxel_um3)

    # map watershed ID -> serial visualization ID
    lys_serial_counts["cell_id_serial"] = (
        lys_serial_counts["cell_id_ch2"]
        .map(cell_id_map_viz)
        .fillna(0)
        .astype(int)
    )

    # assign cell volume
    lys_serial_counts["cell_volume_um3"] = lys_serial_counts["cell_id_serial"].apply(
        lambda sid: float(vol_serial_um3[sid])
        if (sid > 0 and sid < len(vol_serial_um3))
        else 0.0
    )

    lys_serial_counts.to_csv(outpath("lysosome_counts_by_cell_serial.csv"), index=False)

print("Saved:",
      outpath("lysosome_counts_cell_vs_outside.csv"),
      outpath("lysosome_counts_by_cell.csv"),
      outpath("lysosomes_with_cell_vs_outside.csv"),
      outpath("lysosome_counts_by_cell_serial.csv"))

# ==========================================
# SIGNAL + VOLUME QUANTIFICATION 
# ==========================================

print("\n[Signal quantification] Starting...")

signal_img = img_ch1.astype(np.float32)
print(f"Voxel intensities: {signal_img}")

# ---- background subtraction ----
bg = np.percentile(signal_img, 5)

signal_img = np.clip(signal_img - bg, 0, None)
print(f"Subtracts background from every voxel: {signal_img}")

# ---- get valid cell IDs ----
#cell_seg is the watershed segmentation image.
print(f"The watershed segmentation image: {cell_seg}")
cell_ids = np.unique(cell_seg)
print(f"Unique labels: {cell_ids}")
cell_ids = cell_ids[cell_ids > 0]
print(f"Unique labels removing background: {cell_ids}")
print(f"Number of cells: {len(cell_ids)}")

# ---- total signal + volume per cell ----
cell_signal = {}
cell_volume_vox = {}

for cid in cell_ids:

    #cell_mask = (cell_seg == cid)
    cell_mask = (cell_seg == cid) & neuron_mask_for_quantification

    # CELL SIGNAL FROM CH1
    cell_signal[cid] = float(signal_img[cell_mask].sum())

    # CELL VOLUME
    cell_volume_vox[cid] = int(cell_mask.sum())

    print(
        f"[CELL {cid}] "
        f"voxels={cell_volume_vox[cid]} "
        f"signal={cell_signal[cid]}"
    )

# ==========================================
# helper: sphere mask
# ==========================================
def sphere_mask(zc_um, yc_um, xc_um, r_um):

    zc = int(round(zc_um / vz_um))
    yc = int(round(yc_um / vy_um))
    xc = int(round(xc_um / vx_um))

    rz = max(1, int(np.ceil(r_um / vz_um)))
    ry = max(1, int(np.ceil(r_um / vy_um)))
    rx = max(1, int(np.ceil(r_um / vx_um)))

    Z, Y, X = signal_img.shape

    z1, z2 = max(0, zc - rz), min(Z, zc + rz + 1)
    y1, y2 = max(0, yc - ry), min(Y, yc + ry + 1)
    x1, x2 = max(0, xc - rx), min(X, xc + rx + 1)

    zz, yy, xx = np.mgrid[z1:z2, y1:y2, x1:x2]

    dz = (zz - zc) * vz_um
    dy = (yy - yc) * vy_um
    dx = (xx - xc) * vx_um

    sphere = (dz**2 + dy**2 + dx**2) <= (r_um**2)

    return z1, z2, y1, y2, x1, x2, sphere

# ==========================================
# build NON-overlapping lysosome core mask
# ==========================================
lys_mask = np.zeros_like(signal_img, dtype=bool)

for _, row in df.iterrows():

    if row["location_ch2"] != "cell":
        continue

    z1, z2, y1, y2, x1, x2, sphere = sphere_mask(
        row["z_um"],
        row["y_um"],
        row["x_um"],
        row["radius_um"]
    )

    lys_mask[z1:z2, y1:y2, x1:x2] |= sphere

# ==========================================
# Puncta-associated halo mask
# ==========================================
HALO_UM = 0.4  # tune: 0.3-0.8 um

dist_to_lys_um = distance_transform_edt(
    ~lys_mask,
    sampling=(vz_um, vy_um, vx_um)
).astype(np.float32)

lys_assoc_mask = dist_to_lys_um <= HALO_UM
residual_mask = ~lys_assoc_mask

tiff.imwrite(outpath("debug_lysosome_core_mask.tif"), lys_mask.astype(np.uint8) * 255)
tiff.imwrite(outpath("debug_lysosome_associated_mask.tif"), lys_assoc_mask.astype(np.uint8) * 255)
tiff.imwrite(outpath("debug_distance_to_lysosomes_um.tif"), dist_to_lys_um.astype(np.float32))

# ==========================================
# Distance-to-puncta bins
# ==========================================
DISTANCE_BINS_UM = [
    (0.0, 0.2),
    (0.2, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, np.inf),
]


# ==========================================
# Cortical/peripheral zone settings
# ==========================================
CORTEX_UM = 1.0  # tune based on cell size and resolution

# ==========================================
# Compute signal + volume per cell
# ==========================================
rows = []

for cid in cell_ids:

    # Whole cell mask
    cell_mask = (cell_seg == cid)

    # If you want membrane/shell-based cell quantification, keep this:
    cell_quant_mask = (cell_seg == cid) & neuron_mask_for_quantification

    # Total cell signal
    total_signal = float(signal_img[cell_quant_mask].sum())
    total_vol = float(cell_quant_mask.sum() * voxel_um3)

    # -------------------------------
    # Core lysosome signal
    # -------------------------------
    core_mask = cell_mask & lys_mask

    lysosome_core_signal = float(signal_img[core_mask].sum())
    lysosome_core_volume_um3 = float(core_mask.sum() * voxel_um3)

    # -------------------------------
    # Puncta-associated signal
    # core + halo/shadow
    # -------------------------------
    assoc_mask = cell_mask & lys_assoc_mask

    lysosome_assoc_signal = float(signal_img[assoc_mask].sum())
    lysosome_assoc_volume_um3 = float(assoc_mask.sum() * voxel_um3)

    # -------------------------------
    # Residual non-puncta signal
    # outside core + halo
    # -------------------------------
    residual_cell_mask = cell_mask & residual_mask

    residual_signal = float(signal_img[residual_cell_mask].sum())
    residual_volume_um3 = float(residual_cell_mask.sum() * voxel_um3)

    puncta_core_fraction = (
        lysosome_core_signal / total_signal
        if total_signal > 0 else np.nan
    )

    puncta_associated_fraction = (
        lysosome_assoc_signal / total_signal
        if total_signal > 0 else np.nan
    )

    residual_fraction = (
        residual_signal / total_signal
        if total_signal > 0 else np.nan
    )

    # -------------------------------
    # Cortical/peripheral zone
    # -------------------------------
    cell_dist_in_um = distance_transform_edt(
        cell_mask,
        sampling=(vz_um, vy_um, vx_um)
    ).astype(np.float32)

    cortex_mask = cell_mask & (cell_dist_in_um <= CORTEX_UM)
    inner_mask = cell_mask & (cell_dist_in_um > CORTEX_UM)

    cortex_residual_mask = cortex_mask & residual_mask
    inner_residual_mask = inner_mask & residual_mask

    residual_cortex_mean_HA = (
        float(signal_img[cortex_residual_mask].mean())
        if cortex_residual_mask.any() else np.nan
    )

    residual_inner_mean_HA = (
        float(signal_img[inner_residual_mask].mean())
        if inner_residual_mask.any() else np.nan
    )

    residual_cortical_enrichment = (
        residual_cortex_mean_HA / residual_inner_mean_HA
        if np.isfinite(residual_inner_mean_HA) and residual_inner_mean_HA > 0
        else np.nan
    )

    row = {
        "cell_id": int(cid),

        # total cell
        "cell_signal_total": total_signal,
        "cell_volume_um3": total_vol,

        # lysosome core
        "lysosome_core_signal": lysosome_core_signal,
        "lysosome_core_volume_um3": lysosome_core_volume_um3,
        "puncta_core_fraction": puncta_core_fraction,

        # lysosome-associated = core + halo
        "lysosome_associated_signal": lysosome_assoc_signal,
        "lysosome_associated_volume_um3": lysosome_assoc_volume_um3,
        "puncta_associated_fraction": puncta_associated_fraction,

        # residual signal outside puncta-associated region
        "residual_non_puncta_signal": residual_signal,
        "residual_non_puncta_volume_um3": residual_volume_um3,
        "residual_fraction": residual_fraction,

    }

    # -------------------------------
    # 2. Distance-to-puncta bins
    # -------------------------------
    for lo, hi in DISTANCE_BINS_UM:

        if np.isinf(hi):
            bin_mask = cell_mask & (dist_to_lys_um >= lo)
            label_txt = f"{lo:g}_plus"
        else:
            bin_mask = (
                cell_mask &
                (dist_to_lys_um >= lo) &
                (dist_to_lys_um < hi)
            )
            label_txt = f"{lo:g}_{hi:g}"

        label_txt = label_txt.replace(".", "p")

        bin_signal = float(signal_img[bin_mask].sum())
        bin_volume_um3 = float(bin_mask.sum() * voxel_um3)

        row[f"HA_signal_{label_txt}_um_from_puncta"] = bin_signal
        row[f"volume_{label_txt}_um_from_puncta"] = bin_volume_um3

        row[f"mean_HA_{label_txt}_um_from_puncta"] = (
            bin_signal / bin_mask.sum()
            if bin_mask.sum() > 0 else np.nan
        )

    rows.append(row)

    print(
        f"[CELL {cid}] "
        f"total_signal={total_signal:.2f} "
        f"core={lysosome_core_signal:.2f} "
        f"assoc={lysosome_assoc_signal:.2f} "
        f"residual={residual_signal:.2f} "
    )


# ==========================================
# save new signal table
# ==========================================
df_signal = pd.DataFrame(rows)

df_signal.to_csv(
    outpath("cell_signal_vs_lysosome_signal_extended.csv"),
    index=False
)

print(
    "[Signal quantification] Saved:",
    outpath("cell_signal_vs_lysosome_signal_extended.csv")
)

# ==========================================
# Diameter-interval subset (for visualization + interval datasets)
# ==========================================
df_interval = df.copy()
if isinstance(df_interval, pd.DataFrame) and len(df_interval) > 0:
    if DIAMETER_MIN_UM is not None:
        df_interval = df_interval[df_interval["diameter_um"] >= float(DIAMETER_MIN_UM)].copy()
    if DIAMETER_MAX_UM is not None:
        df_interval = df_interval[df_interval["diameter_um"] <= float(DIAMETER_MAX_UM)].copy()

if isinstance(df_interval, pd.DataFrame) and len(df_interval) > 0:
    df_interval["cell_id_ch2_viz"] = df_interval["cell_id_ch2"].map(cell_id_map_viz).fillna(0).astype(int)
    df_interval["lys_id_in_cell"] = 0
    mask_in_i = (df_interval.get("location_ch2", "") == "cell") & (df_interval.get("cell_id_ch2", 0) > 0)
    if np.any(mask_in_i.to_numpy() if hasattr(mask_in_i, "to_numpy") else mask_in_i):
        df_sorted_i = df_interval.loc[mask_in_i].sort_values(["cell_id_ch2", "z_um", "y_um", "x_um"]).copy()
        df_interval.loc[df_sorted_i.index, "lys_id_in_cell"] = (df_sorted_i.groupby("cell_id_ch2").cumcount().to_numpy() + 1).astype(int)

df_interval.to_csv(outpath("lysosomes_with_cell_vs_outside_diameter_interval.csv"), index=False)
print("Saved:", outpath("lysosomes_with_cell_vs_outside_diameter_interval.csv"))

# ==========================================
# Napari edits ONLY the interval subset
# ==========================================
df_for_editing = df_interval.copy()
df = df_for_editing  # your Napari block uses `df`
df_viz = df_for_editing

# ==========================================
# FULL-SIZE OVERLAY EXPORT (same size as raw)  (VISUALIZATION = interval)
# ==========================================
export_fullsize_overlay_stack(
    img_ch1=img_ch1,
    img_ch2_raw=img_ch2,
    cell_seg_viz=cell_seg_viz,
    df=df_viz,
    vx_um=vx_um, vy_um=vy_um, vz_um=vz_um,
    output_dir=output_dir,
    alpha_labels=0.45,
    draw_only_inside=True,
    fps=FPS,
    basename="FULLSIZE_overlay_CellID_Lysosomes_diameter_interval"
)

# ==========================================
# Videos (VISUALIZATION = interval)
# ==========================================
if GENERATE_VIDEOS:
    img_norm_2 = (ch2 * 255).astype(np.uint8)
    frames_fused = []
    Z = img_norm_2.shape[0]
    px_um_xy = float(np.sqrt(vx_um * vy_um))

    for z in range(Z):
        base = cv2.cvtColor(img_norm_2[z], cv2.COLOR_GRAY2BGR)

        cell = (cell_mask_viz[z].astype(np.uint8) * 255)
        overlay = base.copy()
        overlay[..., 1] = np.maximum(overlay[..., 1], cell)
        overlay = cv2.addWeighted(base, 1.0, overlay, 0.35, 0.0)

        drew_any = False

        if isinstance(df_viz, pd.DataFrame) and len(df_viz) > 0:
            dfv = df_viz[
                np.isfinite(df_viz["z_um"]) &
                np.isfinite(df_viz["y_um"]) &
                np.isfinite(df_viz["x_um"]) &
                np.isfinite(df_viz["radius_um"])
            ]
            if not dfv.empty:
                zc = (dfv["z_um"].to_numpy() / vz_um).astype(float)
                yc = (dfv["y_um"].to_numpy() / vy_um).astype(float)
                xc = (dfv["x_um"].to_numpy() / vx_um).astype(float)
                r_um = dfv["radius_um"].to_numpy().astype(float)

                dz_vox = np.abs(zc - z)
                dz_um = dz_vox * vz_um
                hits = dz_um <= r_um

                if np.any(hits):
                    r_proj_um = np.sqrt(np.clip(r_um[hits]**2 - dz_um[hits]**2, 0.0, None))
                    r_proj_vox = r_proj_um / max(px_um_xy, 1e-12)

                    ys = np.rint(yc[hits]).astype(int)
                    xs = np.rint(xc[hits]).astype(int)

                    H, W = cell_mask_viz.shape[1], cell_mask_viz.shape[2]
                    min_radius_px = 3
                    thickness = 2

                    for y, x, rpv in zip(ys, xs, r_proj_vox):
                        rr = int(max(min_radius_px, round(rpv)))
                        if 0 <= y < H and 0 <= x < W and rr > 0:
                            cv2.circle(overlay, (x, y), rr, (0, 0, 0),
                                       thickness + 2, lineType=cv2.LINE_AA)
                            cv2.circle(overlay, (x, y), rr, (255, 255, 0),
                                       thickness, lineType=cv2.LINE_AA)
                    drew_any = True

        if (not drew_any) and (blobs is not None) and (len(blobs) > 0):
            z_blobs = blobs[np.abs(blobs[:, 0] - z) < 0.5]
            H, W = cell_mask_viz.shape[1], cell_mask_viz.shape[2]
            min_radius_px = 3
            thickness = 2
            for b in z_blobs:
                y, x = int(round(b[1])), int(round(b[2]))
                rpx = int(max(min_radius_px, round(b[3])))
                if 0 <= y < H and 0 <= x < W and rpx > 0:
                    cv2.circle(overlay, (x, y), rpx, (0, 0, 0),
                               thickness + 2, lineType=cv2.LINE_AA)
                    cv2.circle(overlay, (x, y), rpx, (255, 255, 0),
                               thickness, lineType=cv2.LINE_AA)

        cv2.putText(overlay, "FUSED (diameter interval + viz mask)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        frames_fused.append(overlay)

    try:
        imageio.mimsave(outpath("ch2_fused_cell_diameter_interval.mp4"), frames_fused, fps=FPS, format="FFMPEG")
        print("Saved:", outpath("ch2_fused_cell_diameter_interval.mp4"))
    except TypeError:
        imageio.mimsave(outpath("ch2_fused_cell_diameter_interval.gif"), frames_fused, fps=FPS)
        print("Saved:", outpath("ch2_fused_cell_diameter_interval.gif"))

    ch1_u8 = _norm_u8_stack(img_ch1.astype(np.float32))
    ch2_u8 = _norm_u8_stack(ch2.astype(np.float32) * 255.0 / max(1.0, ch2.max()))

    frames_raw = []
    frames_fused_all = []
    frames_side_by_side = []

    for z in range(Z):
        b = ch1_u8[z]
        g = ch2_u8[z]
        r = ch1_u8[z]
        base = np.dstack([b, g, r])

        cell = (cell_mask_viz[z].astype(np.uint8) * 255)
        overlay = base.copy()
        overlay[..., 1] = np.maximum(overlay[..., 1], cell)
        overlay = cv2.addWeighted(base, 1.0, overlay, 0.35, 0.0)

        drew_any = False

        if isinstance(df_viz, pd.DataFrame) and len(df_viz) > 0:
            dfv = df_viz[
                np.isfinite(df_viz["z_um"]) &
                np.isfinite(df_viz["y_um"]) &
                np.isfinite(df_viz["x_um"]) &
                np.isfinite(df_viz["radius_um"])
            ]
            if not dfv.empty:
                zc = (dfv["z_um"].to_numpy() / vz_um).astype(float)
                yc = (dfv["y_um"].to_numpy() / vy_um).astype(float)
                xc = (dfv["x_um"].to_numpy() / vx_um).astype(float)
                r_um = dfv["radius_um"].to_numpy().astype(float)

                dz_vox = np.abs(zc - z)
                dz_um = dz_vox * vz_um
                hits = dz_um <= r_um

                if np.any(hits):
                    r_proj_um = np.sqrt(np.clip(r_um[hits]**2 - dz_um[hits]**2, 0.0, None))
                    r_proj_vox = r_proj_um / max(px_um_xy, 1e-12)

                    ys = np.rint(yc[hits]).astype(int)
                    xs = np.rint(xc[hits]).astype(int)

                    H, W = cell_mask_viz.shape[1], cell_mask_viz.shape[2]
                    min_radius_px = 3
                    thickness = 2

                    for y, x, rpv in zip(ys, xs, r_proj_vox):
                        rr = int(max(min_radius_px, round(rpv)))
                        if 0 <= y < H and 0 <= x < W and rr > 0:
                            cv2.circle(overlay, (x, y), rr, (0, 0, 0),
                                       thickness + 2, lineType=cv2.LINE_AA)
                            cv2.circle(overlay, (x, y), rr, (255, 255, 0),
                                       thickness, lineType=cv2.LINE_AA)
                    drew_any = True

        if (not drew_any) and (blobs is not None) and (len(blobs) > 0):
            z_blobs = blobs[np.abs(blobs[:, 0] - z) < 0.5]
            H, W = cell_mask_viz.shape[1], cell_mask_viz.shape[2]
            min_radius_px = 3
            thickness = 2
            for b_ in z_blobs:
                y, x = int(round(b_[1])), int(round(b_[2]))
                rpx = int(max(min_radius_px, round(b_[3])))
                if 0 <= y < H and 0 <= x < W and rpx > 0:
                    cv2.circle(overlay, (x, y), rpx, (0, 0, 0),
                               thickness + 2, lineType=cv2.LINE_AA)
                    cv2.circle(overlay, (x, y), rpx, (255, 255, 0),
                               thickness, lineType=cv2.LINE_AA)

        cv2.putText(base, "RAW (Ch1+Ch2)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, "FUSED (diameter interval + viz mask)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        frames_raw.append(base)
        frames_fused_all.append(overlay)
        frames_side_by_side.append(cv2.hconcat([base, overlay]))

    def _save_video_sync(basename, frames):
        try:
            with imageio.get_writer(
                outpath(f"{basename}.mp4"),
                fps=FPS,
                format="FFMPEG",
                codec="libx264",
                macro_block_size=None
            ) as w:
                for fr in frames:
                    w.append_data(fr)
            print(f"Saved: {outpath(f'{basename}.mp4')} @ {FPS} fps")
        except Exception:
            imageio.mimsave(outpath(f"{basename}.gif"), frames, duration=1.0 / FPS)
            print(f"Saved: {outpath(f'{basename}.gif')} @ {FPS} fps equivalent")

    _save_video_sync("ch2_fused_all_viz_diameter_interval", frames_fused_all)
    _save_video_sync("ch2_raw", frames_raw)
    _save_video_sync("ch2_raw_and_fused_all_viz_diameter_interval", frames_side_by_side)

# ==========================================
# PER-LYSOSOME SIGNAL 
# ==========================================

print("[Per-lysosome signal] Computing...")

signal_img = img_ch1.astype(np.float32) 

# optional background subtraction (recommended)
bg = np.percentile(signal_img, 5)
signal_img = np.clip(signal_img - bg, 0, None)

lys_signal_individual = []

for _, row in df.iterrows():

    z1, z2, y1, y2, x1, x2, sphere = sphere_mask(
        row["z_um"], row["y_um"], row["x_um"], row["radius_um"]
    )

    patch = signal_img[z1:z2, y1:y2, x1:x2]

    val = float(patch[sphere].sum())
    lys_signal_individual.append(val)

df["lysosome_signal_individual"] = np.array(lys_signal_individual, dtype=float)

print("[Per-lysosome signal] Done.")

# ==========================================
# Slice-by-slice overlap visualization:
# Ch1 membrane signal over Ch2-derived 5-voxel mask
# ==========================================

mask_5vox = neuron_mask_for_quantification.astype(bool)

ch1_raw = img_ch1.astype(np.float32)

# Normalize Ch1 for visualization
ch1_u8 = _norm_u8_stack(ch1_raw)

# Threshold Ch1 so "overlap" means real Ch1 signal, not background
ch1_bg = np.percentile(ch1_raw, 5)
ch1_signal_mask = ch1_raw > ch1_bg

# Overlap = Ch1 signal inside the 5-voxel Ch2 mask
overlap_mask = mask_5vox & ch1_signal_mask

Z, H, W = ch1_u8.shape
frames = np.zeros((Z, H, W, 3), dtype=np.uint8)

for z in range(Z):
    # base: Ch1 in grayscale
    base = np.dstack([ch1_u8[z], ch1_u8[z], ch1_u8[z]]).astype(np.float32)

    # blue = Ch2-derived 5-voxel mask
    blue = np.zeros_like(base)
    blue[..., 2] = 255

    # red/yellow = overlap of Ch1 signal with mask
    overlap = np.zeros_like(base)
    overlap[..., 0] = 255
    overlap[..., 1] = 80

    mask_z = mask_5vox[z]
    overlap_z = overlap_mask[z]

    out = base.copy()

    # mask overlay in blue
    out[mask_z] = 0.55 * out[mask_z] + 0.45 * blue[mask_z]

    # overlap overlay in red/yellow, stronger color
    out[overlap_z] = 0.25 * out[overlap_z] + 0.75 * overlap[overlap_z]

    frames[z] = np.clip(out, 0, 255).astype(np.uint8)

# Save as slice-by-slice RGB TIFF
tiff.imwrite(
    outpath("SLICE_BY_SLICE_Ch1_overlap_with_5vox_Ch2_mask.tif"),
    frames,
    photometric="rgb"
)

print("Saved:", outpath("SLICE_BY_SLICE_Ch1_overlap_with_5vox_Ch2_mask.tif"))

# ==========================================
# SIGNAL PERCENTAGE OF CH1 INSIDE MASK PER CELL
# ==========================================

print("\n[Signal in mask per cell] Starting...")

# ---- use CH1 as signal channel ----
signal_img = img_ch1.astype(np.float32)

# ---- background subtraction ----
bg = np.percentile(signal_img, 5)
signal_img = np.clip(signal_img - bg, 0, None)

print(f"[DEBUG] Background CH1 = {bg:.3f}")

# ---- get valid cell IDs ----
cell_ids = np.unique(cell_seg)
cell_ids = cell_ids[cell_ids > 0]

rows = []

for cid in cell_ids:

    # full cell mask for this cell
    cell_mask = (cell_seg == cid) & neuron_mask_for_quantification

    # your membrane/shell mask inside this cell
    mask_in_cell = cell_mask & neuron_mask

    # total CH1 signal inside whole cell
    total_cell_signal = float(signal_img[cell_mask].sum())

    # CH1 signal inside membrane/shell mask
    mask_signal = float(signal_img[mask_in_cell].sum())

    # percentage of CH1 signal falling in the mask
    signal_percent_in_mask = (
        mask_signal / total_cell_signal * 100.0
    ) if total_cell_signal > 0 else 0.0

    rows.append({
        "cell_id": int(cid),
        "cell_signal_total_CH1": total_cell_signal,
        "signal_inside_mask_CH1": mask_signal,
        "signal_percent_inside_mask": signal_percent_in_mask,
    })

    print(
        f"[CELL {cid}] "
        f"total_CH1={total_cell_signal:.2f} | "
        f"mask_CH1={mask_signal:.2f} | "
        f"percent={signal_percent_in_mask:.2f}%"
    )

df_signal_mask = pd.DataFrame(rows)

df_signal_mask.to_csv(
    outpath("cell_CH1_signal_percent_inside_mask.csv"),
    index=False
)

print(
    "[Signal in mask per cell] Saved:",
    outpath("cell_CH1_signal_percent_inside_mask.csv")
)

total_signal_all = signal_img[cell_seg > 0].sum()

mask_signal_all = signal_img[
    (cell_seg > 0) & neuron_mask
].sum()

global_percent = (
    mask_signal_all /
    total_signal_all *
    100
)

print(
    f"Global CH1 signal in mask = "
    f"{global_percent:.2f}%"
)

# ======================================================================
# NAPARI BLOCK 
# ======================================================================

if LAUNCH_VIEWER:
    viewer = napari.Viewer()
    viewer.dims.ndisplay = 3

    viewer.add_image(img_ch2, name="Ch2 raw")
    viewer.add_image(img_ch1, name="Ch1 raw")

    mask_layer = viewer.add_labels(
        cell_mask_viz.astype(np.uint8),
        name="Neurite mask" if NEURITE_MODE else "Cell mask",
        opacity=0.35
    )
    
    try:
        mask_layer.blending = "translucent_no_depth"
    except Exception:
        pass

    n_labels = int(cell_seg_viz.max()) if isinstance(cell_seg_viz, np.ndarray) else 0
    cmap_u8 = make_label_colormap(n_labels, seed_hue=0.13)

    label_color = {0: (0.0, 0.0, 0.0, 0.0)}
    for i in range(1, n_labels + 1):
        r, g, b = cmap_u8[i].astype(np.float32) / 255.0
        label_color[i] = (float(r), float(g), float(b), 1.0)

    def _add_labels_with_color(data, name, opacity=0.25, visible=True):
        try:
            layer = viewer.add_labels(
                data, name=name, opacity=float(opacity), visible=bool(visible), color=label_color
            )
            return layer
        except TypeError:
            layer = viewer.add_labels(data, name=name, opacity=float(opacity), visible=bool(visible))
            try:
                layer.color = label_color
            except Exception:
                pass
            return layer

    id_layer = _add_labels_with_color(cell_seg_viz.astype(np.uint16), "ID (serial)", opacity=0.25, visible=True)
    try:
        id_layer.blending = "translucent_no_depth"
    except Exception:
        pass

    id_filtered_layer = _add_labels_with_color(
        np.zeros_like(cell_seg_viz, dtype=np.uint16),
        "ID (filtered view)",
        opacity=0.45,
        visible=False
    )
    try:
        id_filtered_layer.blending = "translucent_no_depth"
    except Exception:
        pass

    pts_layer = None
    view_pts_layer = None

    serial_to_original_id = None
    if isinstance(cell_id_map_viz, dict) and len(cell_id_map_viz) > 0:
        serial_to_original_id = {serial_id: original_id for (original_id, serial_id) in cell_id_map_viz.items()}

    def _points_rgba_from_props(props_dict):
        loc = np.array(props_dict.get("location_ch2", []), dtype=str)
        cid = np.array(props_dict.get("cell_id_serial", []), dtype=int)

        n = int(len(cid))
        rgba = np.zeros((n, 4), dtype=np.float32)
        rgba[:] = (1.0, 0.0, 1.0, 0.85)

        inside = (loc == "cell") & (cid > 0)
        if np.any(inside):
            for k in np.where(inside)[0]:
                c = int(cid[k])
                rgba[k] = np.array(label_color.get(c, (1.0, 0.0, 1.0, 0.85)), dtype=np.float32)
        return rgba

    def _refresh_point_colors(layer):
        if layer is None:
            return
        p = dict(layer.properties)
        try:
            layer.face_color = _points_rgba_from_props(p)
        except Exception:
            pass

    _filter_state = {
        "cell_ids": None,
        "location": "both",
        "lys_ids": None,
        "show_filtered_labels": True,
        "show_view_layer": True,
    }

    def _parse_id_text(s, max_id):
        s = (s or "").strip().lower()
        if s in ("", "all", "*"):
            return list(range(1, int(max_id) + 1))

        parts = re.split(r"[,\s;]+", s)
        out = set()
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                try:
                    a = int(a)
                    b = int(b)
                except Exception:
                    continue
                lo, hi = (a, b) if a <= b else (b, a)
                for v in range(lo, hi + 1):
                    if 1 <= v <= int(max_id):
                        out.add(v)
            else:
                try:
                    v = int(part)
                except Exception:
                    continue
                if 1 <= v <= int(max_id):
                    out.add(v)
        return sorted(out)

    def _parse_lys_text(s):
        s = (s or "").strip().lower()
        if s in ("", "all", "*"):
            return None
        parts = re.split(r"[,\s;]+", s)
        out = set()
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                try:
                    a = int(a)
                    b = int(b)
                except Exception:
                    continue
                lo, hi = (a, b) if a <= b else (b, a)
                for v in range(lo, hi + 1):
                    if v >= 1:
                        out.add(v)
            else:
                try:
                    v = int(part)
                except Exception:
                    continue
                if v >= 1:
                    out.add(v)
        return sorted(out) if len(out) else None

    def _update_filtered_labels(sel_ids):
        if sel_ids is None or len(sel_ids) == 0:
            id_filtered_layer.visible = False
            id_filtered_layer.data = np.zeros_like(cell_seg_viz, dtype=np.uint16)
            return
        mask = np.isin(cell_seg_viz, np.array(sel_ids, dtype=np.int32))
        out = np.zeros_like(cell_seg_viz, dtype=np.uint16)
        out[mask] = cell_seg_viz[mask].astype(np.uint16)
        id_filtered_layer.data = out
        id_filtered_layer.visible = True

    def _apply_points_filter(sel_ids, location_mode="both", lys_ids=None, show_view_layer=True):
        if pts_layer is None or view_pts_layer is None:
            return
        p = dict(pts_layer.properties)
        loc = np.array(p.get("location_ch2", []), dtype=str)
        cid = np.array(p.get("cell_id_serial", []), dtype=int)
        lys = np.array(p.get("lys_id_serial", []), dtype=int)

        sel_ids = set(int(x) for x in (sel_ids or []))
        keep = np.isin(cid, list(sel_ids))

        if location_mode == "cell":
            keep &= (loc == "cell")
        elif location_mode == "outside":
            keep &= (loc != "cell")

        if lys_ids is not None:
            lys_ids = set(int(x) for x in lys_ids)
            keep &= np.isin(lys, list(lys_ids))

        idx = np.where(keep)[0]

        view_pts_layer.data = np.asarray(pts_layer.data)[idx]
        view_pts_layer.size = np.asarray(pts_layer.size)[idx]
        view_props = {k: np.asarray(v)[idx] for k, v in p.items()}
        view_pts_layer.properties = view_props
        _refresh_point_colors(view_pts_layer)
        view_pts_layer.visible = bool(show_view_layer)

    def _reapply_current_filter():
        if _filter_state["cell_ids"] is None:
            return
        if _filter_state["show_filtered_labels"]:
            _update_filtered_labels(_filter_state["cell_ids"])
        else:
            _update_filtered_labels([])
        _apply_points_filter(
            _filter_state["cell_ids"],
            location_mode=_filter_state["location"],
            lys_ids=_filter_state["lys_ids"],
            show_view_layer=_filter_state["show_view_layer"],
        )

    if EDIT_LYSOSOME_TABLE_IN_NAPARI and isinstance(df, pd.DataFrame) and len(df) > 0:
        df_edit = attach_all_blob_fields(df.copy())
        # ==========================================
        # MAP CELL SIGNAL TO EACH LYSOSOME
        # ==========================================

        if 'df_signal' in globals():

            cell_signal_dict = dict(zip(df_signal["cell_id"], df_signal["cell_signal_total"]))

            # map per lysosome
            df_edit["cell_signal_total"] = df_edit["cell_id_ch2"].map(cell_signal_dict).fillna(0.0)

        else:
            print("[Warning] df_signal missing")
            df_edit["cell_signal_total"] = 0.0

        # ==========================================
        # ADD CELL / SIGNAL FEATURES TO LYSOSOMES TABLE
        # ==========================================
        # Make sure df_signal exists
        if 'df_signal' in globals():

            # Build lookup dictionaries (by original watershed ID)
            cell_signal_dict = dict(zip(df_signal["cell_id"], df_signal["cell_signal_total"]))
            lys_signal_dict  = dict(zip(df_signal["cell_id"], df_signal["lysosome_core_signal"]))
            cell_vol_dict    = dict(zip(df_signal["cell_id"], df_signal["cell_volume_um3"]))
            lys_vol_dict     = dict(zip(df_signal["cell_id"], df_signal["lysosome_core_volume_um3"]))

            # Map values to each lysosome (using ORIGINAL cell IDs)
            df_edit["cell_signal_total"] = df_edit["cell_id_ch2"].map(cell_signal_dict).fillna(0.0)
            df_edit["lysosome_core_signal"]   = df_edit["cell_id_ch2"].map(lys_signal_dict).fillna(0.0)

            df_edit["cell_volume_um3"]   = df_edit["cell_id_ch2"].map(cell_vol_dict).fillna(0.0)
            df_edit["lysosome_core_volume_um3"] = df_edit["cell_id_ch2"].map(lys_vol_dict).fillna(0.0)

        else:
            print("[Warning] df_signal not found — signal features not added.")
            
        pts_zyx = np.stack([
            (df_edit["z_um"].to_numpy() / vz_um),
            (df_edit["y_um"].to_numpy() / vy_um),
            (df_edit["x_um"].to_numpy() / vx_um),
        ], axis=1).astype(np.float32)

        if "radius_um" in df_edit.columns:
            radii_vox = df_edit["radius_um"].to_numpy(dtype=float) / (np.sqrt(vx_um * vy_um) + 1e-12)
            sizes = np.clip(radii_vox * 2, 2, None).astype(np.float32)
        else:
            sizes = np.full((pts_zyx.shape[0],), 6, dtype=np.float32)

        if "location_ch2" not in df_edit.columns:
            df_edit["location_ch2"] = "outside"

        if "cell_id_serial" in df_edit.columns:
            df_edit["cell_id_serial"] = df_edit["cell_id_serial"].fillna(0).astype(int)
        elif "cell_id_ch2_viz" in df_edit.columns:
            df_edit["cell_id_serial"] = df_edit["cell_id_ch2_viz"].fillna(0).astype(int)
        else:
            df_edit["cell_id_serial"] = df_edit["cell_id_ch2"].map(cell_id_map_viz).fillna(0).astype(int)

        if "diameter_um" not in df_edit.columns:
            df_edit["diameter_um"] = np.nan
        if "peak_gray" not in df_edit.columns:
            df_edit["peak_gray"] = np.nan

        def _recompute_lys_id_serial(cell_id_serial_arr, location_arr):
            n = int(len(cell_id_serial_arr))
            out = np.zeros(n, dtype=int)
            mask = (location_arr.astype(str) == "cell") & (cell_id_serial_arr.astype(int) > 0)
            if not np.any(mask):
                return out
            tmp = pd.DataFrame({
                "idx": np.arange(n, dtype=int),
                "cell_id_serial": cell_id_serial_arr.astype(int),
                "z_um": df_edit["z_um"].to_numpy(dtype=float),
                "y_um": df_edit["y_um"].to_numpy(dtype=float),
                "x_um": df_edit["x_um"].to_numpy(dtype=float),
            })
            tmp = tmp.loc[mask].sort_values(["cell_id_serial", "z_um", "y_um", "x_um"], kind="mergesort")
            serial = (tmp.groupby("cell_id_serial").cumcount().to_numpy() + 1).astype(int)
            out[tmp["idx"].to_numpy(dtype=int)] = serial
            return out

        loc0 = df_edit["location_ch2"].astype(str).to_numpy()
        cell_serial0 = df_edit["cell_id_serial"].astype(int).to_numpy()
        lys_serial0 = _recompute_lys_id_serial(cell_serial0, loc0)
        df_edit["lys_id_serial"] = lys_serial0

        props = {
            "location_ch2": loc0,
            "cell_id_serial": cell_serial0,
            "lys_id_serial": lys_serial0,
            "diameter_um": df_edit["diameter_um"].to_numpy(dtype=float),
            "peak_gray": df_edit["peak_gray"].to_numpy(dtype=float),
        }

        pts_layer = viewer.add_points(
            pts_zyx,
            size=sizes,
            name="Lysosomes (EDIT TABLE)",
            properties=props,
        )
        pts_layer.edge_color = "black"
        pts_layer.edge_width = 0.3
        pts_layer.mode = "select"
        pts_layer.text = {"string": "{cell_id_serial}:{lys_id_serial}", "size": 10, "color": "white"}
        _refresh_point_colors(pts_layer)

        view_pts_layer = viewer.add_points(
            pts_zyx[:0],
            size=np.array([], dtype=np.float32),
            name="Lysosomes (VIEW FILTER)",
            properties={k: np.asarray(v)[:0] for k, v in props.items()},
        )
        view_pts_layer.edge_color = "black"
        view_pts_layer.edge_width = 0.2
        view_pts_layer.mode = "pan_zoom"
        view_pts_layer.visible = False
        view_pts_layer.text = {"string": "{cell_id_serial}:{lys_id_serial}", "size": 10, "color": "white"}

        def _refresh_labels_and_serials():
            if pts_layer is None:
                return
            p = dict(pts_layer.properties)
            loc = np.array(p["location_ch2"]).astype(str)
            cell_serial = np.array(p["cell_id_serial"]).astype(int)
            lys_serial = _recompute_lys_id_serial(cell_serial, loc)
            p["lys_id_serial"] = lys_serial
            pts_layer.properties = p
            _refresh_point_colors(pts_layer)
            _reapply_current_filter()

        def _apply_props_update(indices, new_loc=None, new_cell_serial=None):
            if pts_layer is None or not indices:
                return
            p = dict(pts_layer.properties)
            loc = np.array(p["location_ch2"]).astype(object)
            cell_serial = np.array(p["cell_id_serial"]).astype(int)

            for i in indices:
                if new_loc is not None:
                    loc[i] = str(new_loc)
                if new_cell_serial is not None:
                    sid = int(new_cell_serial)
                    cell_serial[i] = sid
                if new_loc == "outside":
                    cell_serial[i] = 0

            p["location_ch2"] = loc.astype(str)
            p["cell_id_serial"] = cell_serial.astype(int)
            pts_layer.properties = p
            _refresh_labels_and_serials()

        @viewer.bind_key("A")
        def assign_selected_to_id_under_cursor(event=None):
            if pts_layer is None:
                return
            sel = sorted(list(pts_layer.selected_data))
            if not sel:
                print("[Napari edit] No points selected.")
                return

            zf, yf, xf = viewer.cursor.position
            zz, yy, xx = int(round(zf)), int(round(yf)), int(round(xf))
            if not (0 <= zz < cell_seg_viz.shape[0] and 0 <= yy < cell_seg_viz.shape[1] and 0 <= xx < cell_seg_viz.shape[2]):
                print("[Napari edit] Cursor out of bounds.")
                return

            serial_id = int(cell_seg_viz[zz, yy, xx])
            if serial_id <= 0:
                print("[Napari edit] Cursor not over a labeled SERIAL ID.")
                return

            _apply_props_update(sel, new_loc="cell", new_cell_serial=serial_id)
            print(f"[Napari edit] Assigned {len(sel)} lysosomes -> serial ID {serial_id}")

        @viewer.bind_key("X")
        def mark_selected_outside(event=None):
            if pts_layer is None:
                return
            sel = sorted(list(pts_layer.selected_data))
            if not sel:
                print("[Napari edit] No points selected.")
                return
            _apply_props_update(sel, new_loc="outside", new_cell_serial=0)
            print(f"[Napari edit] Marked {len(sel)} lysosomes as outside")

        def _export_edited_csv(path):
            if pts_layer is None:
                return
            p = pts_layer.properties
            df_out = df_edit.copy()
            df_out["location_ch2"] = np.array(p["location_ch2"]).astype(str)
            df_out["cell_id_serial"] = np.array(p["cell_id_serial"]).astype(int)
            df_out["lys_id_serial"] = np.array(p["lys_id_serial"]).astype(int)
            df_out = attach_all_blob_fields(df_out)
            df_out.to_csv(outpath(path), index=False)
            print(f"[Napari edit] Saved edited lysosome table: {outpath(path)}")

        @viewer.bind_key("S")
        def save_now(event=None):
            _export_edited_csv(LYSOSOME_EDITED_CSV)

        try:
            from magicgui import magicgui

            @magicgui(
                call_button="Apply filter",
                ids_text={"label": "Cell IDs (e.g. all, 1,2,5-8)", "value": "all"},
                lys_text={"label": "Lys IDs in those cells (e.g. all, 1,2 or 1-3)", "value": "all"},
                location={"choices": ["both", "cell", "outside"], "value": "both"},
                show_filtered_labels={"label": "Show filtered ID labels layer", "value": True},
                show_view_layer={"label": "Show VIEW FILTER points layer", "value": True},
            )
            def filter_panel(ids_text="all", lys_text="all", location="both", show_filtered_labels=True, show_view_layer=True):
                sel_ids = _parse_id_text(ids_text, n_labels)
                lys_ids = _parse_lys_text(lys_text)

                _filter_state["cell_ids"] = sel_ids
                _filter_state["location"] = str(location)
                _filter_state["lys_ids"] = lys_ids
                _filter_state["show_filtered_labels"] = bool(show_filtered_labels)
                _filter_state["show_view_layer"] = bool(show_view_layer)

                _reapply_current_filter()
                msg_lys = "all" if lys_ids is None else str(lys_ids)
                print(f"[Filter] cells={sel_ids} location={location} lys={msg_lys}")

            @magicgui(call_button="Clear filter")
            def clear_filter_panel():
                _filter_state["cell_ids"] = None
                _filter_state["location"] = "both"
                _filter_state["lys_ids"] = None
                _filter_state["show_filtered_labels"] = True
                _filter_state["show_view_layer"] = True

                _update_filtered_labels([])
                if view_pts_layer is not None:
                    view_pts_layer.visible = False
                    view_pts_layer.data = np.zeros((0, 3), dtype=np.float32)
                print("[Filter] Cleared.")

            @magicgui(
                call_button="Select in EDIT layer",
                ids_text={"label": "Cell IDs (e.g. 5 or 1,3-4)", "value": "1"},
                lys_text={"label": "Lys IDs (optional, e.g. all, 1,2)", "value": "all"},
                location={"choices": ["both", "cell", "outside"], "value": "cell"},
            )
            def select_panel(ids_text="1", lys_text="all", location="cell"):
                if pts_layer is None:
                    return
                sel_ids = _parse_id_text(ids_text, n_labels)
                lys_ids = _parse_lys_text(lys_text)

                p = dict(pts_layer.properties)
                loc = np.array(p.get("location_ch2", []), dtype=str)
                cid = np.array(p.get("cell_id_serial", []), dtype=int)
                lys = np.array(p.get("lys_id_serial", []), dtype=int)

                keep = np.isin(cid, sel_ids)
                if location == "cell":
                    keep &= (loc == "cell")
                elif location == "outside":
                    keep &= (loc != "cell")
                if lys_ids is not None:
                    keep &= np.isin(lys, lys_ids)

                idx = np.where(keep)[0]
                pts_layer.selected_data = set(int(i) for i in idx.tolist())
                viewer.layers.selection.active = pts_layer
                msg_lys = "all" if lys_ids is None else str(lys_ids)
                print(f"[Select] Selected {len(idx)} points (cells={sel_ids}, lys={msg_lys}, loc={location}).")

            viewer.window.add_dock_widget(filter_panel, area="right", name="Filter cells / lysosomes")
            viewer.window.add_dock_widget(select_panel, area="right", name="Select lysosomes")
            viewer.window.add_dock_widget(clear_filter_panel, area="right", name="Clear filter")

        except Exception as e:
            print("[Napari] magicgui not available; filter panel disabled. Error:", e)

    try:
        viewer.camera.zoom = 1.2
    except Exception:
        pass

    napari.run()
    
    if EDIT_LYSOSOME_TABLE_IN_NAPARI and pts_layer is not None:
        try:
            p = pts_layer.properties
            df_out = df_edit.copy()
            df_out["location_ch2"] = np.array(p["location_ch2"]).astype(str)
            df_out["cell_id_serial"] = np.array(p["cell_id_serial"]).astype(int)
            df_out["lys_id_serial"] = np.array(p["lys_id_serial"]).astype(int)
            df_out = attach_all_blob_fields(df_out)
            df_out.to_csv(outpath(LYSOSOME_EDITED_CSV), index=False)
            print(f"[Napari edit] Auto-saved edited lysosome table: {outpath(LYSOSOME_EDITED_CSV)}")
        except Exception as e:
            print("[Napari edit] Auto-save failed:", e)

