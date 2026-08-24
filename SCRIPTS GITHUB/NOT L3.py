#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==========================================================
# IMPORT LIBRARIES
# These libraries provide all the functions needed for
# image processing, segmentation, visualization, GUI,
# file management, and numerical computations.
# ==========================================================

# ==========================================================
# Standard Python libraries
# ==========================================================

import os
# Operating system utilities.
# Used to manipulate file paths, folders, and filenames.

import re
# Regular expression library.
# Used for searching or extracting text patterns.

import numpy as np
# Numerical computing library.
# Provides efficient multidimensional arrays and mathematical operations.

import pandas as pd
# Data analysis library.
# Used to create and save tables (CSV files) containing measurements.

import cv2
# OpenCV image processing library.
# Used for filtering, morphology, and video generation.

import imageio
# Library for reading and writing images and videos.

import xml.etree.ElementTree as ET
# XML parser.
# Used to read metadata stored inside CZI image files.

from datetime import datetime
# Provides date and time utilities.
# Used to generate timestamps for output folders and files.

# ==========================================================
# CZI image support
# ==========================================================

import czifile
# Imports the CZI reader for Zeiss microscopy images.

try:
    # Try importing the CZI library.
    import czifile

    # If successful, store no error.
    _CZIFILE_IMPORT_ERROR = None

except Exception as e:

    # If importing fails,
    # disable CZI support.
    czifile = None

    # Save the exception so a meaningful error
    # can be shown later.
    _CZIFILE_IMPORT_ERROR = e

# ==========================================================
# TIFF image support
# ==========================================================

import tifffile as tiff
# Library for reading and writing TIFF microscopy images.

# ==========================================================
# scikit-image (image processing)
# ==========================================================

from skimage.feature import (
    blob_log,           # Laplacian of Gaussian blob detector.
    peak_local_max      # Detects local intensity maxima.
)

from skimage.filters import (
    gaussian,           # Gaussian image smoothing.
    threshold_local     # Adaptive local thresholding.
)

from skimage.morphology import (

    remove_small_objects,   # Removes tiny segmented objects.

    binary_opening,         # Removes small bright artifacts.

    binary_closing,         # Closes small holes and gaps.

    ball,                   # Creates a 3D spherical structuring element.

    binary_erosion,         # Shrinks binary objects.

    binary_dilation,        # Expands binary objects.

    h_maxima                # Finds H-maxima for watershed markers.
)

from skimage.segmentation import watershed
# Watershed segmentation algorithm.

from skimage.measure import label as _label
from skimage.measure import regionprops as _regionprops
from skimage.measure import label
# Labels connected components in binary images.

# ==========================================================
# SciPy (scientific computing)
# ==========================================================

from scipy.ndimage import distance_transform_edt as edt
# Euclidean Distance Transform.
# Imported with the short name "edt".

from scipy.ndimage import distance_transform_edt
# Same function imported with its full name.
# Useful when code prefers the complete function name.

from scipy.ndimage import binary_fill_holes
# Fills holes inside binary objects.

from scipy.optimize import least_squares
# Least-squares optimization algorithm.
# Used for curve or model fitting.

from scipy.ndimage import binary_closing
# Binary closing operation.
# (Same operation also exists in scikit-image.)

from scipy.ndimage import gaussian_filter1d
# One-dimensional Gaussian smoothing.
# Used for smoothing radial intensity profiles.

# ==========================================================
# Visualization
# ==========================================================

import napari
# Interactive multidimensional image viewer.

import PyQt6
# Qt framework used internally by Napari.

import colorsys
# Color conversion utilities.
# Converts between RGB, HSV, HLS, etc.

# ==========================================================
# Graphical User Interface (GUI)
# ==========================================================

import tkinter as tk
# Standard Python GUI toolkit.

from tkinter import (

    ttk,            # Modern Tkinter widgets.

    filedialog,     # File and folder selection dialogs.

    messagebox,     # Popup information and error dialogs.

    simpledialog    # Simple text input dialogs.
)

# ===============================
# GUI (single unified interface)
# ===============================

def get_user_config_gui(
    # ==========================================================
    # Basic parameter defaults
    # These are the values shown when the GUI first opens.
    # ==========================================================

    default_vxy_um=0.04,                 # Default voxel size in the X and Y directions (µm/voxel).

    default_vz_um=None,                  # Default voxel size in the Z direction (µm/voxel).
                                        # If None and the image does not contain Z spacing metadata,
                                        # the program uses the XY voxel size as the fallback.

    default_erode_mult=1.0,              # Multiplier applied to the erosion radius used for
                                        # separating touching lysosomes.

    default_blob_threshold=0.001,        # Detection threshold for the Laplacian of Gaussian
                                        # blob detector. Lower values detect more blobs.

    # ==========================================================
    # Diameter filter defaults
    # Leave as None to disable size filtering.
    # ==========================================================

    default_diam_min_um=None,            # Minimum accepted lysosome diameter (µm).
                                        # None means no lower diameter limit.

    default_diam_max_um=None,            # Maximum accepted lysosome diameter (µm).
                                        # None means no upper diameter limit.

    # ==========================================================
    # Advanced segmentation defaults
    # ==========================================================

    default_margin_um=1.2,               # Maximum distance (µm) from the lysosome core
                                        # used to define the associated region.

    default_overlap_alpha=0.6,           # Fraction of overlap required to associate
                                        # a lysosome with a cell.

    default_neighbor_max_vox=12,         # Maximum neighbor distance (voxels).
                                        # Used internally by the script.
                                        # This parameter is NOT exposed in the GUI.

    default_viz_min_voxels=10000,        # Minimum object size (voxels) displayed
                                        # in some visualization layers.

    default_dist_smooth_sigma=4.0,       # Gaussian smoothing sigma applied to the
                                        # distance transform before watershed.

    default_h_maxima=4.0,                # H-maxima suppression value controlling
                                        # watershed seed generation.

    # ==========================================================
    # Adaptive membrane defaults
    # These control the automatic membrane detection algorithm.
    # ==========================================================

    default_membrane_minimum_bins=8,     # Minimum number of radial bins used to
                                        # build the intensity profile.

    default_membrane_maximum_bins=200,   # Maximum number of radial bins allowed.
                                        # Prevents excessively fine binning.

    default_membrane_sigma_smooth=1.5,   # Gaussian smoothing sigma applied to the
                                        # radial intensity profile before peak detection.

    default_membrane_rel_height=0.6,     # Relative peak height used by peak_widths()
                                        # to estimate membrane thickness.

    # ==========================================================
    # Fixed internal defaults
    # These parameters are not displayed in the GUI.
    # ==========================================================

    default_max_reasonable_vxy_um=0.5,   # Maximum expected XY voxel size (µm).
                                        # Used for metadata validation.

    default_ch1_smooth_sigma=1.0,        # Gaussian smoothing sigma applied
                                        # to channel 1.

    default_blob_min_sigma=0.8,          # Minimum blob size (sigma) for
                                        # Laplacian of Gaussian detection.

    default_blob_max_sigma=2.5,          # Maximum blob size (sigma) for
                                        # Laplacian of Gaussian detection.

    default_blob_num_sigma=12,           # Number of sigma values tested
                                        # between the minimum and maximum.

    default_radial_max_radius_nm=400.0,  # Maximum radius (nm) used when
                                        # computing radial intensity profiles.

    default_radial_dr_nm=5.0,            # Radial sampling interval (nm)
                                        # between consecutive profile points.

    default_radial_min_drop_fraction=0.3,# Minimum intensity decrease required
                                        # to identify the lysosome boundary.

    default_ch2_smooth_sigma=1.5,        # Gaussian smoothing sigma applied
                                        # to channel 2.

    default_thresh_block_size=521,       # Block size used for adaptive thresholding.

    default_thresh_offset_std_mult=0.55, # Standard deviation multiplier used to
                                        # calculate the adaptive threshold offset.

    default_video_fps=8,                 # Frames per second for exported videos.

    # ==========================================================
    # Output options
    # ==========================================================

    default_launch_viewer=True,          # Open Napari automatically when
                                        # processing is finished.

    default_generate_videos=True,        # Automatically generate output videos.
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
    # Dictionary that will store all user-selected parameters.
    # "ok" is initially False and becomes True only if the user
    # presses the Run/OK button and all inputs are valid.
    cfg = {"ok": False}

    # ==========================================================
    # Create the main GUI window
    # ==========================================================

    # Create the main Tkinter window.
    root = tk.Tk()

    # Set the title shown in the window title bar.
    root.title("Lysosome + Cell Segmentation (GUI)")

    # Prevent the user from resizing the GUI window.
    root.resizable(False, False)

    # ==========================================================
    # Variables linked to GUI widgets
    # Each variable is automatically synchronized with its
    # corresponding Entry, Checkbutton, etc.
    # ==========================================================

    # Full path of the selected input image.
    file_var = tk.StringVar(value="")

    # Output directory where all results will be saved.
    out_var = tk.StringVar(value="")

    # ==========================================================
    # Basic segmentation parameters
    # ==========================================================

    # Erosion multiplier used during lysosome segmentation.
    erode_var = tk.StringVar(value=str(default_erode_mult))

    # Blob detector threshold.
    # Lower values detect more candidate lysosomes.
    blob_var = tk.StringVar(value=str(default_blob_threshold))

    # ==========================================================
    # Diameter filter (optional)
    # Leave blank to disable the filter.
    # ==========================================================

    # Minimum accepted lysosome diameter (µm).
    diam_min_var = tk.StringVar(
        value="" if default_diam_min_um is None else str(default_diam_min_um)
    )

    # Maximum accepted lysosome diameter (µm).
    diam_max_var = tk.StringVar(
        value="" if default_diam_max_um is None else str(default_diam_max_um)
    )

    # ==========================================================
    # Advanced parameters
    # ==========================================================

    # Controls whether the Advanced section is expanded or hidden.
    show_adv = tk.BooleanVar(value=False)

    # Maximum distance (µm) defining the associated lysosome region.
    margin_var = tk.StringVar(value=str(default_margin_um))

    # Minimum overlap fraction between lysosome and cell.
    overlap_var = tk.StringVar(value=str(default_overlap_alpha))

    # Minimum object size (voxels) displayed in visualization layers.
    vizmin_var = tk.StringVar(value=str(default_viz_min_voxels))

    # Gaussian smoothing sigma applied to the distance transform
    # before watershed segmentation.
    distsmooth_var = tk.StringVar(value=str(default_dist_smooth_sigma))

    # H-maxima value used to generate watershed seeds.
    hmax_var = tk.StringVar(value=str(default_h_maxima))

    # ==========================================================
    # Adaptive membrane parameters
    # ==========================================================

    # Minimum number of radial bins allowed when building
    # the membrane intensity profile.
    membrane_min_bins_var = tk.StringVar(
        value=str(default_membrane_minimum_bins)
    )

    # Maximum number of radial bins allowed.
    membrane_max_bins_var = tk.StringVar(
        value=str(default_membrane_maximum_bins)
    )

    # Gaussian smoothing sigma applied to the radial profile.
    membrane_sigma_var = tk.StringVar(
        value=str(default_membrane_sigma_smooth)
    )

    # Relative peak height used by peak_widths() to estimate
    # membrane thickness.
    membrane_rel_height_var = tk.StringVar(
        value=str(default_membrane_rel_height)
    )

    # ==========================================================
    # Output options
    # ==========================================================

    # Frames per second for exported videos.
    fps_var = tk.StringVar(value=str(default_video_fps))

    # Whether Napari should automatically open after processing.
    launch_viewer_var = tk.BooleanVar(
        value=bool(default_launch_viewer)
    )

    # Whether output videos should be generated automatically.
    gen_videos_var = tk.BooleanVar(
        value=bool(default_generate_videos)
    )

    # ==========================================================
    # Helper functions used by the GUI
    # These functions manage file selection, validation,
    # and the behavior of the interface.
    # ==========================================================

    def _suggest_output_dir(fp):
        # Automatically generate a default output folder
        # based on the selected image location.

        # If no input file has been selected,
        # return an empty string.
        if not fp:
            return ""

        # Extract the directory containing the input image.
        raw_dir = os.path.dirname(fp)

        # Create a unique timestamp so each analysis
        # produces a separate output folder.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Return the complete output directory path.
        # Example:
        # C:\Images\outputs_20260730_142530
        return os.path.join(raw_dir, f"outputs_{stamp}")


    def browse_file():
        # Open a dialog that lets the user choose
        # the image to analyze.

        fp = filedialog.askopenfilename(

            # Title displayed in the file dialog.
            title="Select image file",

            # Restrict visible files to supported image formats.
            filetypes=[
                ("Image files", "*.tif *.tiff *.czi"),
                ("All files", "*.*"),
            ],
        )

        # Continue only if the user selected a file.
        if fp:

            # Store the selected file path in the GUI.
            file_var.set(fp)

            # If the output folder is still empty,
            # automatically generate a suggested one.
            if not out_var.get().strip():
                out_var.set(_suggest_output_dir(fp))


    def browse_output_dir():
        # Open a dialog that lets the user choose
        # the parent output directory.

        d = filedialog.askdirectory(
            title="Select output folder"
        )

        # Continue only if the user selected a folder.
        if d:

            # Get the currently selected image.
            fp = file_var.get().strip()

            # If an image has already been selected,
            # create a timestamped output folder inside
            # the chosen directory.
            if fp:

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                out_var.set(
                    os.path.join(d, f"outputs_{stamp}")
                )

            # Otherwise simply use the selected directory.
            else:
                out_var.set(d)


    def _err(msg):
        # Display an error dialog with the given message.
        messagebox.showerror(
            "Invalid input",
            msg
        )

        # Stop execution immediately.
        raise ValueError(msg)


    def _float_required(s, name):
        # Read a required floating-point parameter.

        # Remove leading/trailing spaces.
        s = (s or "").strip()

        # Reject empty values.
        if s == "":
            _err(f"{name} is required.")

        try:
            # Convert comma decimal separators into dots
            # before converting to float.
            return float(s.replace(",", "."))

        except Exception:
            # Display a clear error if conversion fails.
            _err(
                f"{name} must be a number (got: {s})"
            )


    def _float_optional(s, name):
        # Read an optional floating-point parameter.

        s = (s or "").strip()

        # Empty value means the parameter is disabled.
        if s == "":
            return None

        try:
            # Convert to floating-point number.
            return float(s.replace(",", "."))

        except Exception:
            _err(
                f"{name} must be a number (got: {s})"
            )


    def _int_required(s, name):
        # Read a required integer parameter.

        s = (s or "").strip()

        # Reject empty values.
        if s == "":
            _err(f"{name} is required.")

        try:
            # Convert to integer.
            # Using int(float(...)) allows inputs like
            # "12.0" while still storing an integer.
            return int(float(s.replace(",", ".")))

        except Exception:
            _err(
                f"{name} must be an integer (got: {s})"
            )

    def _toggle_adv():
        # Show or hide the Advanced Parameters section.
        # If the checkbox is checked,
        # display the Advanced frame.
        if show_adv.get():
            adv_frame.grid()

        # Otherwise hide it while preserving
        # the layout configuration.
        else:
            adv_frame.grid_remove()

    def run_clicked():
        # ==========================================================
        # Callback function executed when the user presses
        # the "Run" button.
        #
        # This function:
        #   1. Reads every value entered in the GUI.
        #   2. Validates all inputs.
        #   3. Stores the configuration in the cfg dictionary.
        #   4. Closes the GUI if everything is valid.
        # ==========================================================

        # ----------------------------------------------------------
        # Read input file
        # ----------------------------------------------------------

        # Get the selected image path from the GUI.
        fp = file_var.get().strip()

        # Ensure an image was selected.
        if not fp:
            _err("Please select a file.")

        # Verify that the selected file actually exists.
        if not os.path.isfile(fp):
            _err("Selected file does not exist.")

        # ----------------------------------------------------------
        # Determine output directory
        # ----------------------------------------------------------

        # Use the user-selected output folder.
        # If empty, automatically generate one.
        outd = out_var.get().strip() or _suggest_output_dir(fp)

        # ==========================================================
        # Read basic parameters
        # ==========================================================

        # Read erosion multiplier.
        erode = _float_required(
            erode_var.get(),
            "ERODE_MULT"
        )

        # Read blob detection threshold.
        blobt = _float_required(
            blob_var.get(),
            "blob_log threshold"
        )

        # Read output video frame rate.
        fps = _int_required(
            fps_var.get(),
            "video FPS"
        )

        # ==========================================================
        # Read optional diameter filter
        # ==========================================================

        # Minimum accepted lysosome diameter.
        dmin = _float_optional(
            diam_min_var.get(),
            "Min lysosome diameter (µm)"
        )

        # Maximum accepted lysosome diameter.
        dmax = _float_optional(
            diam_max_var.get(),
            "Max lysosome diameter (µm)"
        )

        # Minimum diameter cannot be negative.
        if (dmin is not None) and (dmin < 0):
            _err(
                "Min lysosome diameter must be >= 0."
            )

        # Maximum diameter cannot be negative.
        if (dmax is not None) and (dmax < 0):
            _err(
                "Max lysosome diameter must be >= 0."
            )

        # Minimum diameter must be smaller than maximum.
        if (
            (dmin is not None)
            and
            (dmax is not None)
            and
            (dmin > dmax)
        ):
            _err(
                "Min lysosome diameter cannot be larger than Max lysosome diameter."
            )

        # ==========================================================
        # Read Advanced parameters
        # ==========================================================

        # Width of the associated lysosome region.
        margin = _float_required(
            margin_var.get(),
            "MARGIN_UM (µm)"
        )

        # Cell overlap fraction.
        overlap = _float_required(
            overlap_var.get(),
            "OVERLAP_ALPHA (0..1)"
        )

        # Minimum visualization object size.
        vizmin = _int_required(
            vizmin_var.get(),
            "VIZ_MIN_VOXELS (voxels)"
        )

        # Gaussian smoothing sigma for watershed.
        dist_smooth_sigma = _float_required(
            distsmooth_var.get(),
            "DIST_SMOOTH_SIGMA"
        )

        # H-maxima parameter.
        h_maxima = _float_required(
            hmax_var.get(),
            "H_MAXIMA"
        )

        # ==========================================================
        # Read Adaptive membrane parameters
        # ==========================================================

        # Minimum number of radial bins.
        membrane_minimum_bins = _int_required(
            membrane_min_bins_var.get(),
            "Adaptive membrane minimum bins"
        )

        # Maximum number of radial bins.
        membrane_maximum_bins = _int_required(
            membrane_max_bins_var.get(),
            "Adaptive membrane maximum bins"
        )

        # Gaussian smoothing sigma applied to
        # the radial intensity profile.
        membrane_sigma_smooth = _float_required(
            membrane_sigma_var.get(),
            "Adaptive membrane smoothing sigma"
        )

        # Relative height used by peak_widths().
        membrane_rel_height = _float_required(
            membrane_rel_height_var.get(),
            "Adaptive membrane relative height"
        )

        # ==========================================================
        # Validate Advanced parameters
        # ==========================================================

        # Overlap must be between 0 and 1.
        if not (0.0 <= overlap <= 1.0):
            _err(
                "OVERLAP_ALPHA must be between 0 and 1."
            )

        # At least three bins are needed.
        if membrane_minimum_bins < 3:
            _err(
                "Adaptive membrane minimum bins must be at least 3."
            )

        # Maximum bins cannot be smaller than minimum bins.
        if membrane_maximum_bins < membrane_minimum_bins:
            _err(
                "Adaptive membrane maximum bins must be greater than or equal to minimum bins."
            )

        # Sigma must be non-negative.
        if membrane_sigma_smooth < 0:
            _err(
                "Adaptive membrane smoothing sigma must be >= 0."
            )

        # Relative height must lie between 0 and 1.
        if not (0.0 <= membrane_rel_height <= 1.0):
            _err(
                "Adaptive membrane relative height must be between 0 and 1."
            )

        # ==========================================================
        # Store every validated parameter
        # ==========================================================

        # Save all parameters into the configuration dictionary.
        # The rest of the pipeline reads values from this dictionary.
        cfg.update({

            # Analysis accepted.
            "ok": True,

            # Input image path.
            "file_path": fp,

            # Output directory.
            "output_dir": outd,

            # ---------------- Basic parameters ----------------

            "ERODE_MULT": float(erode),

            "BLOB_THRESHOLD": float(blobt),

            # ---------------- Diameter filter ----------------

            "DIAMETER_MIN_UM":
                None if dmin is None else float(dmin),

            "DIAMETER_MAX_UM":
                None if dmax is None else float(dmax),

            # ---------------- Image metadata ----------------

            "DEFAULT_VX_VY_UM":
                float(default_vxy_um),

            "DEFAULT_VZ_UM":
                None if default_vz_um is None else float(default_vz_um),

            # Internal metadata sanity limit.
            "MAX_REASONABLE_VXY_UM":
                float(default_max_reasonable_vxy_um),

            # ---------------- Advanced parameters ----------------

            "MARGIN_UM":
                float(margin),

            "OVERLAP_ALPHA":
                float(overlap),

            "VIZ_MIN_VOXELS":
                int(vizmin),

            "DIST_SMOOTH_SIGMA":
                float(dist_smooth_sigma),

            "H_MAXIMA":
                float(h_maxima),

            # ---------------- Adaptive membrane ----------------

            "MEMBRANE_MINIMUM_BINS":
                int(membrane_minimum_bins),

            "MEMBRANE_MAXIMUM_BINS":
                int(membrane_maximum_bins),

            "MEMBRANE_SIGMA_SMOOTH":
                float(membrane_sigma_smooth),

            "MEMBRANE_REL_HEIGHT":
                float(membrane_rel_height),

            # ---------------- Internal parameters ----------------

            "NEIGHBOR_MAX_VOX":
                int(default_neighbor_max_vox),

            "CH1_SMOOTH_SIGMA":
                float(default_ch1_smooth_sigma),

            "BLOB_MIN_SIGMA":
                float(default_blob_min_sigma),

            "BLOB_MAX_SIGMA":
                float(default_blob_max_sigma),

            "BLOB_NUM_SIGMA":
                int(default_blob_num_sigma),

            "RADIAL_MAX_RADIUS_NM":
                float(default_radial_max_radius_nm),

            "RADIAL_DR_NM":
                float(default_radial_dr_nm),

            "RADIAL_MIN_DROP_FRACTION":
                float(default_radial_min_drop_fraction),

            "CH2_SMOOTH_SIGMA":
                float(default_ch2_smooth_sigma),

            "THRESH_BLOCK_SIZE":
                int(default_thresh_block_size),

            "THRESH_OFFSET_STD_MULT":
                float(default_thresh_offset_std_mult),

            # ---------------- Output options ----------------

            "VIDEO_FPS":
                int(fps),

            "LAUNCH_VIEWER":
                bool(launch_viewer_var.get()),

            "GENERATE_VIDEOS":
                bool(gen_videos_var.get()),
        })

        # ==========================================================
        # Close the GUI
        # ==========================================================

        # If execution reaches this point,
        # all parameters are valid.
        # Close the GUI and continue with the analysis.
        root.destroy()

    # ==========================================================
    # GUI control functions
    # ==========================================================

    def cancel_clicked():
        # Function executed when the user presses the Cancel button.
        # Simply closes the GUI without saving any parameters.
        root.destroy()


    # Register the function that will be called when the user
    # clicks the window close button (X).
    root.protocol("WM_DELETE_WINDOW", cancel_clicked)

    # ==========================================================
    # Main GUI layout
    # ==========================================================

    # Default padding used throughout the GUI.
    pad = {"padx": 10, "pady": 6}

    # Create the main frame that will contain all widgets.
    frm = ttk.Frame(root)

    # Place the frame inside the main window.
    frm.grid(row=0, column=0, sticky="nsew", **pad)

    # ==========================================================
    # Row counter
    # Used to place widgets one row after another.
    # ==========================================================

    r = 0

    # ==========================================================
    # Input image selection
    # ==========================================================

    # Label describing the input image field.
    ttk.Label(frm, text="Image file:").grid(
        row=r,
        column=0,
        sticky="w"
    )

    # Text box displaying the selected image path.
    ttk.Entry(
        frm,
        textvariable=file_var,
        width=60
    ).grid(
        row=r,
        column=1,
        sticky="we"
    )

    # Button that opens the file selection dialog.
    ttk.Button(
        frm,
        text="Browse...",
        command=browse_file
    ).grid(
        row=r,
        column=2,
        sticky="e"
    )

    # Move to the next row.
    r += 1

    # ==========================================================
    # Output folder selection
    # ==========================================================

    # Label for the output folder.
    ttk.Label(frm, text="Output folder:").grid(
        row=r,
        column=0,
        sticky="w"
    )

    # Text box displaying the output directory.
    ttk.Entry(
        frm,
        textvariable=out_var,
        width=60
    ).grid(
        row=r,
        column=1,
        sticky="we"
    )

    # Button that lets the user choose the output folder.
    ttk.Button(
        frm,
        text="Browse...",
        command=browse_output_dir
    ).grid(
        row=r,
        column=2,
        sticky="e"
    )

    r += 1

    # ==========================================================
    # Separator
    # ==========================================================

    # Horizontal line separating sections.
    ttk.Separator(frm).grid(
        row=r,
        column=0,
        columnspan=3,
        sticky="we",
        pady=8
    )

    r += 1

    # ==========================================================
    # Basic parameters
    # ==========================================================

    # Erosion multiplier.
    ttk.Label(frm, text="ERODE_MULT:").grid(
        row=r,
        column=0,
        sticky="w"
    )

    ttk.Entry(
        frm,
        textvariable=erode_var,
        width=20
    ).grid(
        row=r,
        column=1,
        sticky="w"
    )

    ttk.Label(
        frm,
        text="(unitless)"
    ).grid(
        row=r,
        column=2,
        sticky="w"
    )

    r += 1

    # Blob detection threshold.
    ttk.Label(
        frm,
        text="blob_log threshold:"
    ).grid(
        row=r,
        column=0,
        sticky="w"
    )

    ttk.Entry(
        frm,
        textvariable=blob_var,
        width=20
    ).grid(
        row=r,
        column=1,
        sticky="w"
    )

    ttk.Label(
        frm,
        text="(unitless)"
    ).grid(
        row=r,
        column=2,
        sticky="w"
    )

    r += 1

    # ==========================================================
    # Separator
    # ==========================================================

    ttk.Separator(frm).grid(
        row=r,
        column=0,
        columnspan=3,
        sticky="we",
        pady=8
    )

    r += 1

    # ==========================================================
    # Optional diameter filter
    # ==========================================================

    # Minimum lysosome diameter.
    ttk.Label(
        frm,
        text="Min lysosome diameter (µm):"
    ).grid(
        row=r,
        column=0,
        sticky="w"
    )

    ttk.Entry(
        frm,
        textvariable=diam_min_var,
        width=20
    ).grid(
        row=r,
        column=1,
        sticky="w"
    )

    ttk.Label(
        frm,
        text="blank = no filter"
    ).grid(
        row=r,
        column=2,
        sticky="w"
    )

    r += 1

    # Maximum lysosome diameter.
    ttk.Label(
        frm,
        text="Max lysosome diameter (µm):"
    ).grid(
        row=r,
        column=0,
        sticky="w"
    )

    ttk.Entry(
        frm,
        textvariable=diam_max_var,
        width=20
    ).grid(
        row=r,
        column=1,
        sticky="w"
    )

    ttk.Label(
        frm,
        text="blank = no filter"
    ).grid(
        row=r,
        column=2,
        sticky="w"
    )

    r += 1

    # ==========================================================
    # Separator
    # ==========================================================

    ttk.Separator(frm).grid(
        row=r,
        column=0,
        columnspan=3,
        sticky="we",
        pady=8
    )

    r += 1

    # ==========================================================
    # Output options
    # ==========================================================

    # Checkbox controlling whether Napari opens automatically.
    ttk.Checkbutton(
        frm,
        text="Launch Napari viewer",
        variable=launch_viewer_var
    ).grid(
        row=r,
        column=0,
        columnspan=2,
        sticky="w"
    )

    # Checkbox controlling video generation.
    ttk.Checkbutton(
        frm,
        text="Generate videos",
        variable=gen_videos_var
    ).grid(
        row=r,
        column=2,
        sticky="w"
    )

    r += 1

    # Video frame rate.
    ttk.Label(
        frm,
        text="Video FPS:"
    ).grid(
        row=r,
        column=0,
        sticky="w"
    )

    ttk.Entry(
        frm,
        textvariable=fps_var,
        width=20
    ).grid(
        row=r,
        column=1,
        sticky="w"
    )

    ttk.Label(
        frm,
        text="(frames/sec)"
    ).grid(
        row=r,
        column=2,
        sticky="w"
    )

    r += 1

    # ==========================================================
    # Separator
    # ==========================================================

    ttk.Separator(frm).grid(
        row=r,
        column=0,
        columnspan=3,
        sticky="we",
        pady=8
    )

    r += 1

    # ==========================================================
    # Advanced section toggle
    # ==========================================================

    # Checkbox used to show or hide the Advanced settings.
    ttk.Checkbutton(
        frm,
        text="Show advanced settings",
        variable=show_adv,
        command=_toggle_adv
    ).grid(
        row=r,
        column=0,
        columnspan=3,
        sticky="w"
    )

    r += 1

    # ==========================================================
    # Advanced settings frame
    # ==========================================================

    # Create a collapsible frame containing advanced parameters.
    adv_frame = ttk.LabelFrame(
        frm,
        text="Advanced"
    )

    # Place the frame.
    adv_frame.grid(
        row=r,
        column=0,
        columnspan=3,
        sticky="we",
        pady=6
    )

    # Hide the frame initially.
    adv_frame.grid_remove()

    # ==========================================================
    # Advanced row counter
    # ==========================================================

    rr = 0

    def add_row(lbl, var, hint):
        """
        Creates one parameter row inside the
        Advanced settings frame.

        Parameters
        ----------
        lbl : str
        Parameter label.

        var : tk.StringVar
            Variable linked to the Entry widget.

        hint : str
            Small explanation displayed beside the Entry.
        """

        nonlocal rr

        # Parameter label.
        ttk.Label(
            adv_frame,
            text=lbl
        ).grid(
            row=rr,
            column=0,
            sticky="w",
            padx=8,
            pady=3
        )

        # Parameter input box.
        ttk.Entry(
            adv_frame,
            textvariable=var,
            width=18
        ).grid(
            row=rr,
            column=1,
            sticky="w",
            padx=8,
            pady=3
        )

        # Help text describing the parameter.
        ttk.Label(
            adv_frame,
            text=hint
        ).grid(
            row=rr,
            column=2,
            sticky="w",
            padx=8,
            pady=3
        )

        # Move to the next row.
        rr += 1

    # ==========================================================
    # Populate Advanced settings
    # ==========================================================

    add_row(
        "MARGIN_UM:",
        margin_var,
        "µm (soft band around neuron mask)"
    )

    add_row(
        "OVERLAP_ALPHA:",
        overlap_var,
        "unitless (0..1)"
    )

    add_row(
        "VIZ_MIN_VOXELS:",
        vizmin_var,
        "voxels (hide small cells)"
    )

    add_row(
        "DIST_SMOOTH_SIGMA:",
        distsmooth_var,
        "watershed distance-map smoothing"
    )

    add_row(
        "H_MAXIMA:",
        hmax_var,
        "watershed seed detection"
    )

    # Continue placing widgets below the Advanced frame.
    r += 1

    # ==========================================================
    # Adaptive membrane section
    # ==========================================================

    # Create a separate frame containing all adaptive
    # membrane parameters.
    membrane_frame = ttk.LabelFrame(
        frm,
        text="Adaptive membrane"
    )

    # Place the frame below the Advanced section.
    membrane_frame.grid(
        row=r,
        column=0,
        columnspan=3,
        sticky="we",
        pady=6
    )

    # Row counter used inside the Adaptive membrane frame.
    mr = 0

    def add_membrane_row(lbl, var, hint):
        nonlocal mr
        ttk.Label(membrane_frame, text=lbl).grid(
            row=mr, column=0, sticky="w", padx=8, pady=3
        )
        ttk.Entry(membrane_frame, textvariable=var, width=18).grid(
            row=mr, column=1, sticky="w", padx=8, pady=3
        )
        ttk.Label(membrane_frame, text=hint).grid(
            row=mr, column=2, sticky="w", padx=8, pady=3
        )
        mr += 1

    add_membrane_row(
        "Minimum bins:", membrane_min_bins_var,
        "minimum radial bins; integer >= 3"
    )
    add_membrane_row(
        "Maximum bins:", membrane_max_bins_var,
        "maximum radial bins; integer >= minimum"
    )
    add_membrane_row(
        "Smoothing sigma:", membrane_sigma_var,
        "Gaussian smoothing of radial profile; >= 0"
    )
    add_membrane_row(
        "Relative height:", membrane_rel_height_var,
        "peak width level; value from 0 to 1"
    )

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

# ==========================================================
# METADATA PARSING
#
# This section extracts the physical voxel dimensions
# from TIFF/OME metadata or Zeiss CZI metadata.
#
# Returned voxel sizes are expressed in micrometers:
#   X = horizontal voxel size
#   Y = vertical voxel size
#   Z = distance between image slices
# ==========================================================


def _parse_ome_xml(xml_text):
    # Extract voxel dimensions from OME-XML metadata.

    # If no XML metadata was provided, return missing values
    # for the X, Y, and Z voxel sizes.
    if not xml_text:
        return None, None, None

    def grab(attr):
        # Search the XML text for an attribute such as:
        # PhysicalSizeX="0.04"
        # PhysicalSizeY="0.04"
        # PhysicalSizeZ="0.25"

        m = re.search(
            fr'PhysicalSize{attr}="([\d\.eE+-]+)"',
            xml_text
        )

        # Convert the matched value to float.
        # Return None if the attribute was not found.
        return float(m.group(1)) if m else None

    # Extract and return X, Y, and Z voxel sizes.
    return grab("X"), grab("Y"), grab("Z")


def _parse_czi_scaling(czi_text):
    """
    Extract physical voxel dimensions from Zeiss CZI metadata.

    The parser:
    - Supports values stored inside <Value> elements.
    - Supports values stored as XML attributes.
    - Supports XML namespaces.
    - Converts meters or nanometers to micrometers.
    - Uses a regular-expression fallback if XML parsing fails.
    """

    # If no CZI metadata was provided, return missing values.
    if not czi_text:
        return None, None, None

    # Some CZI readers may return metadata as bytes.
    if isinstance(czi_text, (bytes, bytearray)):

        # Decode bytes into a normal Python string.
        # Invalid characters are ignored.
        czi_text = czi_text.decode(
            "utf-8",
            errors="ignore"
        )

    # Remove null characters that can make XML parsing fail.
    czi_text = czi_text.replace("\x00", "")

    def _to_float(s):
        # Safely convert a metadata value to float.

        # Missing input produces a missing result.
        if s is None:
            return None

        try:
            # Remove spaces and support comma decimal separators.
            return float(
                str(s).strip().replace(",", ".")
            )

        except Exception:
            # Invalid numeric text is treated as missing.
            return None

    def _to_um(val, unit_hint=None):
        # Convert a physical distance to micrometers.

        # Missing values remain missing.
        if val is None:
            return None

        # Use the explicit unit from the metadata when available.
        if unit_hint:

            # Normalize the unit text.
            u = str(unit_hint).strip().lower()

            # Convert meters to micrometers.
            if u in (
                "m",
                "meter",
                "metre",
                "meters",
                "metres",
            ):
                return val * 1e6

            # Value is already expressed in micrometers.
            if u in (
                "µm",
                "um",
                "micron",
                "microns",
                "micrometer",
                "micrometre",
            ):
                return val

            # Convert nanometers to micrometers.
            if u in (
                "nm",
                "nanometer",
                "nanometre",
                "nanometers",
                "nanometres",
            ):
                return val / 1000.0

        # ------------------------------------------------------
        # Unit heuristic
        #
        # This is used only when the metadata does not explicitly
        # specify the unit.
        # ------------------------------------------------------

        # Very small values are probably expressed in meters.
        # Example: 4e-8 m = 0.04 µm.
        if val < 1e-3:
            return val * 1e6

        # Values below 10 are assumed to already be micrometers.
        if val < 10:
            return val

        # Larger reasonable values are assumed to be nanometers.
        if val < 1e5:
            return val / 1000.0

        # Extremely large values are considered invalid.
        return None

    try:
        # Parse the CZI metadata as XML.
        root = ET.fromstring(czi_text)

    except Exception:
        # ------------------------------------------------------
        # Fallback parser
        #
        # If XML parsing fails, search the raw text using
        # regular expressions.
        # ------------------------------------------------------

        def _grab(axis):
            # Search for a metadata block similar to:
            #
            # <Distance Id="X">
            #     <Value>4e-8</Value>
            # </Distance>

            mm = re.search(
                rf'<Distance[^>]*Id="{axis}"[^>]*>'
                rf'.*?<Value>\s*([0-9eE\+\-\.]+)\s*</Value>',
                czi_text,
                flags=re.IGNORECASE | re.DOTALL,
            )

            # Convert the matched text into a float.
            return _to_float(mm.group(1)) if mm else None

        # Extract X, Y, and Z values and convert them to µm.
        return (
            _to_um(_grab("X")),
            _to_um(_grab("Y")),
            _to_um(_grab("Z")),
        )

    # Initialize the three voxel dimensions as missing.
    sx = sy = sz = None

    # Find every XML element named Distance.
    # The {*} wildcard allows any XML namespace.
    for d in root.findall(".//{*}Distance"):

        # Try several possible attribute names used to identify
        # the spatial axis.
        axis = (
            d.attrib.get("Id")
            or d.attrib.get("id")
            or d.attrib.get("Axis")
            or d.attrib.get("axis")
        )

        # Ignore Distance elements without an axis identifier.
        if not axis:
            continue

        # Normalize the axis name to uppercase.
        axis = axis.upper()

        # Read the unit if it is stored as an attribute.
        unit = (
            d.attrib.get("Unit")
            or d.attrib.get("unit")
        )

        # First, try to read the physical value directly
        # from the Distance element attributes.
        valf = _to_float(
            d.attrib.get("Value")
            or d.attrib.get("value")
        )

        # If no attribute value exists, search for a nested
        # <Value> element.
        if valf is None:

            # Find a descendant element named Value.
            v_el = d.find(".//{*}Value")

            # Convert its text when available.
            if v_el is not None and v_el.text:
                valf = _to_float(v_el.text)

        # If the value is still missing, inspect all descendant
        # elements in the Distance block.
        if valf is None:

            for child in d.iter():

                # Skip the Distance element itself.
                if child is d:
                    continue

                # Detect elements whose tag ends with "value",
                # regardless of namespace or capitalization.
                if str(child.tag).lower().endswith("value"):

                    # Try reading the value from an attribute first.
                    # If that fails, try the element text.
                    valf = (
                        _to_float(
                            child.attrib.get("Value")
                            or child.attrib.get("value")
                        )
                        or _to_float(child.text)
                    )

                    # Stop searching after finding a valid value.
                    if valf is not None:
                        break

        # Convert the extracted physical value to micrometers.
        val_um = _to_um(
            valf,
            unit_hint=unit
        )

        # Ignore invalid or unavailable values.
        if val_um is None:
            continue

        # Store the value according to its spatial axis.
        if axis == "X":
            sx = val_um

        elif axis == "Y":
            sy = val_um

        elif axis == "Z":
            sz = val_um

    # Return X, Y, and Z voxel dimensions in micrometers.
    return sx, sy, sz

def load_any(file_path):
    """
    Load a two-channel TIFF or CZI microscopy image.

    Returns
    -------
    ch1 : ndarray
        First image channel.

    ch2 : ndarray
        Second image channel.

    voxel_sizes : tuple
        Physical voxel dimensions:
        (vx_um, vy_um, vz_um)

    metadata : dict
        Basic information about the loaded file type.
    """

    # Extract the file extension and convert it to lowercase.
    ext = os.path.splitext(file_path)[1].lower()

    # ======================================================
    # Load TIFF files
    # ======================================================

    if ext in (".tif", ".tiff"):

        # Open the TIFF file safely.
        # The file is closed automatically after this block.
        with tiff.TiffFile(file_path) as tf:

            # Read the complete image array.
            arr = tf.asarray()

            try:
                # Try to read OME-XML metadata.
                ome_xml = tf.ome_metadata

            except Exception:
                # Some TIFF files do not contain OME metadata.
                ome_xml = None

            # Initialize voxel dimensions as unavailable.
            vx_um = vy_um = vz_um = None

            # Parse the voxel dimensions when OME metadata exists.
            if ome_xml:
                vx_um, vy_um, vz_um = _parse_ome_xml(
                    ome_xml
                )

        # Remove array dimensions whose size is 1.
        # Example:
        # (1, 2, 20, 512, 512) -> (2, 20, 512, 512)
        img = np.squeeze(arr)

        # The script expects a four-dimensional two-channel image.
        if img.ndim == 4:

            # Case 1:
            # Channel is the first dimension.
            # Shape example: (2, Z, Y, X)
            if img.shape[0] == 2:
                ch1, ch2 = img[0], img[1]

            # Case 2:
            # Channel is the second dimension.
            # Shape example: (Z, 2, Y, X)
            elif img.shape[1] == 2:
                ch1, ch2 = img[:, 0], img[:, 1]

            # Case 3:
            # Channel is the final dimension.
            # Shape example: (Z, Y, X, 2)
            elif img.shape[-1] == 2:
                ch1, ch2 = img[..., 0], img[..., 1]

            # A four-dimensional array was found, but no
            # dimension has exactly two channels.
            else:
                raise RuntimeError(
                    "Unexpected TIFF shape for 2 channels"
                )

        # The TIFF array does not have the expected number
        # of dimensions.
        else:
            raise RuntimeError(
                "Unexpected TIFF shape"
            )

        # Return both channels, voxel dimensions, and file type.
        return (
            ch1,
            ch2,
            (vx_um, vy_um, vz_um),
            {"type": "tiff"},
        )

    # ======================================================
    # Load Zeiss CZI files
    # ======================================================

    if ext == ".czi":

        # Open the CZI file safely.
        with czifile.CziFile(file_path) as cf:

            # Read the complete image array.
            arr = cf.asarray()

            try:
                # Extract CZI XML metadata.
                czi_xml = cf.metadata()

            except Exception:
                # Continue even when metadata extraction fails.
                czi_xml = None

        # Initialize voxel dimensions as unavailable.
        vx_um = vy_um = vz_um = None

        # Parse physical voxel dimensions from CZI metadata.
        if czi_xml:
            vx_um, vy_um, vz_um = _parse_czi_scaling(
                czi_xml
            )

        # Remove all dimensions whose length is 1.
        img = np.squeeze(arr)

        # The script expects a four-dimensional two-channel image.
        if img.ndim == 4:

            # Channel is the first dimension:
            # (2, Z, Y, X)
            if img.shape[0] == 2:
                ch1, ch2 = img[0], img[1]

            # Channel is the second dimension:
            # (Z, 2, Y, X)
            elif img.shape[1] == 2:
                ch1, ch2 = img[:, 0], img[:, 1]

            # Channel is the final dimension:
            # (Z, Y, X, 2)
            elif img.shape[-1] == 2:
                ch1, ch2 = img[..., 0], img[..., 1]

            # No dimension contains exactly two channels.
            else:
                raise RuntimeError(
                    "Unexpected CZI shape for 2 channels"
                )

        # The CZI array does not have the expected number
        # of dimensions after squeezing.
        else:
            raise RuntimeError(
                "Unexpected CZI shape"
            )

        # Return both image channels, voxel dimensions,
        # and the detected file type.
        return (
            ch1,
            ch2,
            (vx_um, vy_um, vz_um),
            {"type": "czi"},
        )

    # ======================================================
    # Unsupported file type
    # ======================================================

    # Raise an error for formats other than TIFF or CZI.
    raise ValueError(
        "Unsupported file format"
    )

def refine_radii_via_3d_gaussian_fit(

    img3d,                  # 3D image containing the lysosome signal.
                            # Shape = (Z, Y, X).

    blobs,                  # Array of detected blobs.
                            # Each row contains:
                            # [z_center, y_center, x_center, radius]

    vx_um,                  # Physical voxel size along X (µm).

    vy_um,                  # Physical voxel size along Y (µm).

    vz_um,                  # Physical voxel size along Z (µm).

    win_um=0.5,             # Half-width of the cube (µm) extracted
                            # around every blob for Gaussian fitting.
):
    """
    Fit a 3D Gaussian to the intensity distribution around
    every detected blob.

    The fitted Gaussian widths (sigmaX and sigmaY) are used
    to estimate a more accurate blob radius.

    The returned blob array is identical to the input except
    that column 3 contains the refined XY sigma expressed
    in pixels.
    """

    # If there are no detected blobs,
    # simply return the input.
    if blobs is None or len(blobs) == 0:
        return blobs

    # Convert the image to float.
    # Numerical optimization should always use floating point.
    img = img3d.astype(np.float32)

    # Image dimensions.
    Z, Y, X = img.shape

    # Create a copy of the blob table.
    # The original detections remain unchanged.
    out = blobs.copy().astype(np.float32)

    # ----------------------------------------------------------
    # Compute the fitting window size in voxels.
    #
    # Example:
    #
    # win_um = 0.5 µm
    # voxel size = 0.05 µm
    #
    # -> 10 voxels in each direction.
    # ----------------------------------------------------------

    rz = int(np.ceil(win_um / vz_um))    # Half-window in Z voxels.
    ry = int(np.ceil(win_um / vy_um))    # Half-window in Y voxels.
    rx = int(np.ceil(win_um / vx_um))    # Half-window in X voxels.

    # Compute one representative XY voxel size.
    # Used later to convert sigma (µm)
    # into sigma (pixels).
    px_um_xy = float(np.sqrt(vx_um * vy_um))

    # ==========================================================
    # Process every detected blob independently.
    # ==========================================================

    for i, (zc, yc, xc, _) in enumerate(out):

        # Convert the blob center coordinates
        # to integer voxel indices.
        z0 = int(round(zc))
        y0 = int(round(yc))
        x0 = int(round(xc))

        # Ignore blobs that somehow fall
        # outside the image.
        if not (0 <= z0 < Z and 0 <= y0 < Y and 0 <= x0 < X):
            continue

        # ------------------------------------------------------
        # Compute the cube limits.
        # max() and min() prevent indexing
        # outside the image.
        # ------------------------------------------------------

        z1 = max(0, z0 - rz)
        z2 = min(Z, z0 + rz + 1)

        y1 = max(0, y0 - ry)
        y2 = min(Y, y0 + ry + 1)

        x1 = max(0, x0 - rx)
        x2 = min(X, x0 + rx + 1)

        # Extract the local cube.
        patch = img[z1:z2, y1:y2, x1:x2]

        # Skip extremely small cubes.
        # The Gaussian fit would not be reliable.
        if patch.size < 50:
            continue

        # ------------------------------------------------------
        # Generate voxel coordinates.
        # ------------------------------------------------------

        # Create a coordinate grid having
        # exactly the same size as the cube.
        zz, yy, xx = np.mgrid[z1:z2, y1:y2, x1:x2]

        # Convert voxel coordinates into
        # physical distances (µm)
        # relative to the blob center.

        dz = (zz - z0) * vz_um
        dy = (yy - y0) * vy_um
        dx = (xx - x0) * vx_um

        # Flatten the coordinates into an
        # N × 3 matrix.
        #
        # Column 0 = Z distance
        # Column 1 = Y distance
        # Column 2 = X distance
        coords = np.stack(
            [dz.ravel(), dy.ravel(), dx.ravel()],
            axis=1
        )

        # Flatten image intensities into
        # a one-dimensional vector.
        intens = patch.ravel()

        # ------------------------------------------------------
        # Initial parameter guesses.
        # ------------------------------------------------------

        # Initial amplitude.
        A0 = intens.max() - intens.min()

        # Initial background intensity.
        B0 = intens.min()

        # Initial Gaussian width along X (µm).
        sx0 = 0.2

        # Initial Gaussian width along Y (µm).
        sy0 = 0.2

        # Initial Gaussian width along Z (µm).
        sz0 = 0.2

        # Initial parameter vector.
        p0 = np.array([
            A0,
            B0,
            sx0,
            sy0,
            sz0
        ])

        # ------------------------------------------------------
        # Residual function.
        #
        # least_squares() repeatedly calls this
        # function while searching for the
        # best Gaussian parameters.
        # ------------------------------------------------------

        def residuals(p):

            # Extract the current parameters.
            A, B, sx, sy, sz = p

            # Reject impossible Gaussian widths.
            if sx <= 0 or sy <= 0 or sz <= 0:
                return np.ones_like(intens) * 1e6

            # Evaluate the 3D Gaussian model.
            model = A * np.exp(

                # Z contribution.
                -(coords[:, 0]**2) / (2 * sz**2)

                # Y contribution.
                -(coords[:, 1]**2) / (2 * sy**2)

                # X contribution.
                -(coords[:, 2]**2) / (2 * sx**2)

            ) + B

            # Return residuals.
            # The optimizer tries to make these
            # values as close to zero as possible.
            return model - intens

        # ------------------------------------------------------
        # Perform nonlinear optimization.
        # ------------------------------------------------------

        try:

            # Optimize the Gaussian parameters.
            res = least_squares(
                residuals,
                p0,
                method="trf"
            )

            # Read the optimized parameters.
            A, B, sx, sy, sz = res.x

            # --------------------------------------------------
            # Estimate one representative XY sigma.
            # --------------------------------------------------

            # Average the X and Y Gaussian widths.
            sigma_xy_um = (
                0.5 *
                (abs(sx) + abs(sy))
            )

            # Alternative scaling factors that were tested.
            #
            # sigma_xy_um = 0.25 * (abs(sx) + abs(sy))
            # sigma_xy_um = 0.40 * (abs(sx) + abs(sy))
            # sigma_xy_um = 0.60 * (abs(sx) + abs(sy))

            # Convert from micrometers
            # to pixel-equivalent sigma.
            out[i, 3] = sigma_xy_um / px_um_xy

        # Ignore blobs for which the optimization fails.
        except Exception:
            continue

    # Return the refined blob table.
    return out

def refine_radii_via_radial_intensity(
    img3d,                      # 3D microscopy image with shape (Z, Y, X).
    blobs,                      # Blob array: [z_center, y_center, x_center, radius_px].
    vx_um,                      # Physical voxel size along X, in micrometers.
    vy_um,                      # Physical voxel size along Y, in micrometers.
    vz_um,                      # Physical voxel size along Z, in micrometers.
    max_radius_nm=400,          # Maximum radial distance analyzed around each blob, in nm.
                                # 400 nm = 0.4 µm.
    dr_nm=5.0,                  # Width of each radial bin, in nm.
                                # Smaller values produce a finer radial profile.
    min_drop_fraction=0.3,      # Intended minimum intensity drop.
                                # Important: currently this parameter is not used.
):
    """
    Refine the radius of every detected blob using its
    three-dimensional radial intensity profile.

    The function:

    1. Extracts a local region around each blob.
    2. Calculates the physical distance of each voxel
       from the blob center.
    3. Groups voxels into radial-distance bins.
    4. Calculates the mean intensity in each radial bin.
    5. Smooths the radial intensity profile.
    6. Finds the profile peak.
    7. Estimates the full width at half maximum, FWHM.
    8. Uses half of the FWHM as the new blob radius.

    Returns
    -------
    blobs_out : numpy.ndarray
        Copy of the original blob array in which column 3
        contains the refined radius expressed in XY pixels.
    """

    # Check whether the blob array is missing or empty.
    if blobs is None or len(blobs) == 0:

        # Nothing can be processed, so return the original input.
        return blobs

    # Convert the image to 32-bit floating point.
    # Floating-point data are necessary for averaging,
    # filtering, and intensity calculations.
    img = img3d.astype(np.float32)

    # Read the image dimensions.
    #
    # Z = number of optical slices.
    # Y = image height.
    # X = image width.
    Z, Y, X = img.shape

    # Make a floating-point copy of the blob array.
    # This prevents modification of the original blob detections.
    blobs_out = blobs.copy().astype(np.float32)

    # Convert the maximum analysis radius
    # from nanometers to micrometers.
    #
    # Example:
    # 400 nm / 1000 = 0.4 µm.
    max_r_um = max_radius_nm / 1000.0

    # Convert radial-bin width from nanometers
    # to micrometers.
    #
    # Example:
    # 5 nm / 1000 = 0.005 µm.
    dr_um = dr_nm / 1000.0

    # Create the boundaries of the radial bins.
    #
    # For max_r_um = 0.4 and dr_um = 0.005:
    #
    # 0.000, 0.005, 0.010, ..., 0.400 µm.
    r_edges = np.arange(
        0.0,
        max_r_um + dr_um,
        dr_um
    )

    # At least two edges are required to define one interval.
    if r_edges.size < 2:

        # Return the unchanged blob array if no valid
        # radial intervals can be created.
        return blobs_out

    # Calculate the center of every radial bin.
    #
    # Example:
    # A bin from 0.000 to 0.005 µm has its center at 0.0025 µm.
    r_centers = 0.5 * (
        r_edges[:-1] + r_edges[1:]
    )

    # Calculate one representative XY pixel size.
    #
    # The geometric mean is used because X and Y
    # voxel sizes may be slightly different.
    px_um_xy = float(
        np.sqrt(vx_um * vy_um)
    )

    # ==========================================================
    # Process each detected blob independently.
    # ==========================================================

    # i is the row index in blobs_out.
    #
    # zc, yc, xc are the detected blob-center coordinates.
    #
    # r_px_init is the original radius in pixels.
    for i, (zc, yc, xc, r_px_init) in enumerate(blobs_out):

        # Round the Z center to the nearest integer voxel.
        z0 = int(round(zc))

        # Round the Y center to the nearest integer voxel.
        y0 = int(round(yc))

        # Round the X center to the nearest integer voxel.
        x0 = int(round(xc))

        # Check that the blob center is inside the image.
        if not (
            0 <= z0 < Z
            and 0 <= y0 < Y
            and 0 <= x0 < X
        ):

            # Skip this blob if its center is outside the image.
            continue

        # Calculate how many Z voxels are needed
        # to cover max_r_um.
        #
        # max(1, ...) guarantees at least one Z voxel.
        rz = max(
            1,
            int(np.ceil(max_r_um / vz_um))
        )

        # Calculate how many Y voxels are needed
        # to cover max_r_um.
        ry = max(
            1,
            int(np.ceil(max_r_um / vy_um))
        )

        # Calculate how many X voxels are needed
        # to cover max_r_um.
        rx = max(
            1,
            int(np.ceil(max_r_um / vx_um))
        )

        # Calculate the first Z index of the local patch.
        #
        # max(0, ...) prevents a negative index.
        z1 = max(
            0,
            z0 - rz
        )

        # Calculate the last exclusive Z index.
        #
        # min(Z, ...) prevents indexing beyond the image.
        z2 = min(
            Z,
            z0 + rz + 1
        )

        # Calculate the first Y index of the local patch.
        y1 = max(
            0,
            y0 - ry
        )

        # Calculate the last exclusive Y index.
        y2 = min(
            Y,
            y0 + ry + 1
        )

        # Calculate the first X index of the local patch.
        x1 = max(
            0,
            x0 - rx
        )

        # Calculate the last exclusive X index.
        x2 = min(
            X,
            x0 + rx + 1
        )

        # Verify that the patch has a valid size
        # in all three dimensions.
        if z1 >= z2 or y1 >= y2 or x1 >= x2:

            # Skip invalid or empty patches.
            continue

        # Extract the local 3D region around the blob.
        patch = img[
            z1:z2,
            y1:y2,
            x1:x2
        ]

        # Create coordinate grids for all voxels
        # inside the local patch.
        #
        # zz contains Z coordinates.
        # yy contains Y coordinates.
        # xx contains X coordinates.
        zz, yy, xx = np.mgrid[
            z1:z2,
            y1:y2,
            x1:x2
        ]

        # Calculate each voxel's Z distance from the blob center.
        #
        # Multiplying by vz_um converts voxels to micrometers.
        dz_um = (
            zz - z0
        ) * vz_um

        # Calculate each voxel's Y distance from the blob center
        # in micrometers.
        dy_um = (
            yy - y0
        ) * vy_um

        # Calculate each voxel's X distance from the blob center
        # in micrometers.
        dx_um = (
            xx - x0
        ) * vx_um

        # Calculate the physical 3D Euclidean distance
        # from every voxel to the blob center.
        #
        # r = sqrt(z² + y² + x²)
        r_um = np.sqrt(
            dz_um**2
            + dy_um**2
            + dx_um**2
        )

        # Create a spherical mask that keeps only voxels
        # inside the selected maximum radius.
        mask = (
            r_um <= max_r_um
        )

        # Check whether the spherical mask contains any voxels.
        if not np.any(mask):

            # Skip this blob if the mask is empty.
            continue

        # Extract radial distances inside the spherical mask.
        #
        # ravel() converts the result into a one-dimensional array.
        r_vals = r_um[
            mask
        ].ravel()

        # Extract the image intensity of the same voxels.
        I_vals = patch[
            mask
        ].ravel()

        # Assign each radial distance to one radial bin.
        #
        # np.digitize() returns indices beginning at 1.
        #
        # Subtracting 1 converts them into zero-based indices.
        bin_idx = np.digitize(
            r_vals,
            r_edges
        ) - 1

        # Identify bin indices that lie inside
        # the available radial-profile range.
        valid = (
            (bin_idx >= 0)
            & (bin_idx < r_centers.size)
        )

        # Check whether at least one valid bin exists.
        if not np.any(valid):

            # Skip the current blob if no voxel
            # was assigned to a valid bin.
            continue

        # Keep only valid radial-bin indices.
        bin_idx = bin_idx[
            valid
        ]

        # Keep only intensities corresponding
        # to valid radial-bin indices.
        I_vals = I_vals[
            valid
        ]

        # Calculate the total intensity in every radial bin.
        #
        # weights=I_vals means that intensities,
        # rather than voxel counts, are accumulated.
        sums = np.bincount(
            bin_idx,
            weights=I_vals,
            minlength=r_centers.size
        )

        # Count how many voxels belong to every radial bin.
        counts = np.bincount(
            bin_idx,
            minlength=r_centers.size
        )

        # Temporarily suppress warnings caused by
        # invalid divisions or divisions by zero.
        with np.errstate(
            invalid="ignore",
            divide="ignore"
        ):

            # Calculate the mean intensity of every radial bin.
            #
            # np.maximum(counts, 1) avoids division by zero.
            prof = sums / np.maximum(
                counts,
                1
            )

        # Identify radial bins that contain
        # at least one image voxel.
        have = (
            counts > 0
        )

        # Check whether any nonempty radial bin exists.
        if not np.any(have):

            # Skip the current blob if every bin is empty.
            continue

        # Keep the radius values for nonempty bins.
        r_prof = r_centers[
            have
        ]

        # Keep the mean-intensity values for nonempty bins
        # and convert them to float32.
        I_prof = prof[
            have
        ].astype(np.float32)

        # ------------------------------------------------------
        # Smooth the radial intensity profile.
        # ------------------------------------------------------

        # Apply one-dimensional Gaussian smoothing
        # to reduce noise in the radial intensity profile.
        #
        # sigma=1.0 is measured in profile bins.
        I_smooth = gaussian_filter1d(
            I_prof,
            sigma=1.0
        )

        # Find the maximum intensity in the smoothed profile.
        I_max = float(
            I_smooth.max()
        )

        # Find the minimum intensity in the smoothed profile.
        I_min = float(
            I_smooth.min()
        )

        # Reject profiles without a positive intensity maximum.
        if I_max <= 0:

            # Skip this blob.
            continue

        # Find the index of the maximum intensity.
        #
        # This is the radial-profile peak.
        peak_idx = int(
            np.argmax(I_smooth)
        )

        # Count the total number of radial bins
        # in the smoothed profile.
        n_bins = len(
            I_smooth
        )

        # ------------------------------------------------------
        # Estimate intensity at the left side of the peak.
        # ------------------------------------------------------

        # Start the left-side search at the peak.
        left_probe = peak_idx

        # Move from the peak toward smaller radial distances.
        #
        # The loop continues while the current intensity
        # is greater than or equal to the global minimum.
        while (
            left_probe > 0
            and I_smooth[left_probe] >= I_min
        ):
            left_probe -= 1

        # ------------------------------------------------------
        # Estimate intensity at the right side of the peak.
        # ------------------------------------------------------

        # Start the right-side search at the peak.
        right_probe = peak_idx

        # Move from the peak toward larger radial distances.
        while (
            right_probe < n_bins - 1
            and I_smooth[right_probe] >= I_min
        ):
            right_probe += 1

        # Read the intensity at the selected left position.
        I_left = float(
            I_smooth[left_probe]
        )

        # Read the intensity at the selected right position.
        I_right = float(
            I_smooth[right_probe]
        )

        # ------------------------------------------------------
        # Evaluate profile symmetry.
        # ------------------------------------------------------

        # Calculate the normalized difference
        # between the left and right intensities.
        #
        # A value close to zero means that both sides
        # have similar intensity.
        #
        # max(I_max, 1e-9) prevents division by zero.
        symmetry_ratio = (
            abs(I_left - I_right)
            / max(I_max, 1e-9)
        )

        # Select the baseline used for the
        # half-maximum calculation.
        if symmetry_ratio == 0:

            # If the two sides are exactly equal,
            # treat the profile as symmetric.
            #
            # Use the global minimum as baseline.
            baseline = I_min

        else:

            # For an asymmetric profile,
            # use the higher side intensity as baseline.
            baseline = max(
                I_left,
                I_right
            )

        # Calculate the intensity halfway between
        # the selected baseline and the peak.
        #
        # This is the half-maximum threshold.
        I_half = (
            baseline
            + 0.5 * (
                I_max - baseline
            )
        )

        # ------------------------------------------------------
        # Find the left FWHM crossing.
        # ------------------------------------------------------

        # Begin at the profile peak.
        left_idx = peak_idx

        # Move left while the intensity remains
        # at or above the half-maximum threshold.
        while (
            left_idx > 0
            and I_smooth[left_idx] >= I_half
        ):
            left_idx -= 1

        # If the loop moved one position below the threshold,
        # return one bin toward the peak.
        if (
            left_idx < peak_idx
            and I_smooth[left_idx] < I_half
        ):
            left_idx += 1

        # ------------------------------------------------------
        # Find the right FWHM crossing.
        # ------------------------------------------------------

        # Begin at the profile peak.
        right_idx = peak_idx

        # Move right while the intensity remains
        # at or above the half-maximum threshold.
        while (
            right_idx < n_bins - 1
            and I_smooth[right_idx] >= I_half
        ):
            right_idx += 1

        # If the loop moved one position below the threshold,
        # return one bin toward the peak.
        if (
            right_idx > peak_idx
            and I_smooth[right_idx] < I_half
        ):
            right_idx -= 1

        # ------------------------------------------------------
        # Validate the FWHM interval.
        # ------------------------------------------------------

        # The right crossing must be located
        # after the left crossing.
        if right_idx <= left_idx:

            # Skip invalid FWHM measurements.
            continue

        # Convert the left crossing index
        # into a physical radial position in micrometers.
        r_left = float(
            r_prof[left_idx]
        )

        # Convert the right crossing index
        # into a physical radial position in micrometers.
        r_right = float(
            r_prof[right_idx]
        )

        # Reject undefined numerical values.
        if np.isnan(r_left) or np.isnan(r_right):

            # Skip this blob.
            continue

        # Verify that the right position
        # is greater than the left position.
        if r_right <= r_left:

            # Skip zero-width or reversed intervals.
            continue

        # ------------------------------------------------------
        # Calculate radius from the FWHM.
        # ------------------------------------------------------

        # Full width at half maximum:
        #
        # FWHM = r_right - r_left
        #
        # This code defines the radius as:
        #
        # radius = FWHM / 2
        radius_um = 0.5 * (
            r_right - r_left
        )

        # Reject zero or negative radius measurements.
        if radius_um <= 0:

            # Skip this blob.
            continue

        # Convert the radius from micrometers
        # to equivalent XY pixels.
        #
        # The 1e-9 safeguard prevents division by zero.
        r_fwhm_px = (
            radius_um
            / max(px_um_xy, 1e-9)
        )

        # ------------------------------------------------------
        # Option A: true radius replacement.
        # ------------------------------------------------------

        # Replace the original blob radius
        # with the radius measured from the radial profile.
        #
        # This option can make the radius either
        # larger or smaller than the original radius.
        blobs_out[i, 3] = float(
            r_fwhm_px
        )

        # ------------------------------------------------------
        # Option B: only permit radius growth.
        # ------------------------------------------------------

        # To use this option, comment Option A above
        # and uncomment the following line.
        #
        # The radius will never become smaller
        # than the initial blob radius.
        #
        # blobs_out[i, 3] = max(
        #     float(r_px_init),
        #     float(r_fwhm_px)
        # )

    # Return the blob array with the refined radius
    # stored in column 3.
    return blobs_out

# ==========================================================
# AUTOMATIC MORPHOLOGY
#
# This section automatically estimates suitable radii for:
#   1. Binary opening
#   2. Binary closing
#   3. Binary erosion
#
# The estimated radii are based on:
#   - the internal distance transform of the neuron mask
#   - the sizes of connected components in the neuron mask
# ==========================================================
def _equiv_radius_from_area(px_area):
    """
    Convert an area into the radius of a circle
    having the same area.

    Parameters
    ----------
    px_area : float
        Area measured in pixels or voxels.

    Returns
    -------
    float
        Equivalent circular radius.
    """

    # Make sure the supplied area is at least 1.0.
    #
    # This prevents:
    #   - division problems
    #   - square-root problems
    #   - a completely zero equivalent radius
    safe_area = max(
        px_area,
        1.0
    )

    # For a circle:
    #
    # area = pi * radius^2
    #
    # Rearranging gives:
    #
    # radius = sqrt(area / pi)
    radius = np.sqrt(
        safe_area / np.pi
    )

    # Convert the NumPy result into a normal Python float.
    return float(radius)


def _component_size_percentile(
    mask_bool,
    pct=0.2
):
    """
    Calculate a selected percentile of connected-component sizes.

    Parameters
    ----------
    mask_bool : ndarray
        Boolean binary mask.

    pct : float
        Percentile written as a fraction from 0 to 1.

        Examples:
            0.20 = 20th percentile
            0.50 = median
            0.90 = 90th percentile

    Returns
    -------
    float
        Connected-component area at the requested percentile.
    """

    # Label all connected foreground components.
    #
    # Background pixels or voxels receive label 0.
    # Each connected foreground object receives a unique
    # positive integer label.
    lab = _label(
        mask_bool
    )

    # Build a list containing the area of every
    # foreground connected component.
    #
    # r.area:
    #   Returns the number of pixels or voxels in the component.
    #
    # r.label != 0:
    #   Excludes the background label.
    sizes = [
        r.area
        for r in _regionprops(lab)
        if r.label != 0
    ]

    # Check whether the mask contains no connected objects.
    if not sizes:

        # Return zero when no component sizes are available.
        return 0.0

    # Sort component sizes from smallest to largest.
    #
    # Then convert the Python list into a NumPy array.
    sizes = np.array(
        sorted(sizes)
    )

    # Convert the percentile fraction into a percentage.
    #
    # Example:
    # pct = 0.20 becomes 20.0.
    percentile_number = (
        pct * 100.0
    )

    # Calculate the requested percentile of component sizes.
    selected_size = np.percentile(
        sizes,
        percentile_number
    )

    # Return the result as a normal Python float.
    return float(selected_size)

def auto_morphology_params(
    neuron_mask,          # Boolean neuron segmentation mask.
    vx_um,                # Voxel size along X, in micrometers.
    vy_um,                # Voxel size along Y, in micrometers.
    vz_um,                # Voxel size along Z, in micrometers.
    p_open=2,             # Multiplier for estimating opening radius.
    p_close=1.8,          # Multiplier for estimating closing radius.
    p_erode=0.15,         # Multiplier for estimating erosion radius.
    min_r_open=1,         # Minimum opening radius, in voxels.
    min_r_close=1,        # Minimum closing radius, in voxels.
    max_r=12,             # Maximum permitted morphology radius.
):
    """
    Automatically estimate opening, closing, and erosion radii.

    The opening radius is based on the size of relatively
    small connected components.

    The closing and erosion radii are based on the internal
    Euclidean distance transform of the neuron mask.

    Returns
    -------
    tuple
        Three integer values:

        (
            opening_radius,
            closing_radius,
            erosion_radius
        )
    """

    # Calculate the Euclidean distance transform
    # of the neuron mask.
    #
    # For every foreground voxel, din contains the distance
    # to the nearest background voxel.
    #
    # Background positions normally contain zero.
    din = edt(
        neuron_mask
    )

    # Check whether the neuron mask contains
    # at least one foreground voxel.
    if np.any(neuron_mask):

        # Select distance-transform values only
        # inside the neuron mask.
        internal_distances = din[
            neuron_mask
        ]

        # Calculate the median internal distance.
        #
        # This gives a robust estimate of the typical
        # distance from the interior of the object
        # to its nearest boundary.
        r_in_med = float(
            np.median(
                internal_distances
            )
        )

    else:

        # If the mask is empty, use zero.
        r_in_med = 0.0

    # Calculate the 20th percentile of connected-component areas.
    #
    # This provides a representative size for relatively
    # small foreground components.
    small_px = _component_size_percentile(
        neuron_mask,
        pct=0.20
    )

    # Convert the selected small-component area
    # into an equivalent circular radius.
    r_small = _equiv_radius_from_area(
        small_px
    )

    # Estimate the closing radius.
    #
    # The typical internal distance is multiplied by p_close.
    estimated_close = round(
        p_close * r_in_med
    )

    # Restrict the closing radius to the interval:
    #
    # min_r_close <= r_close <= max_r
    r_close = int(
        np.clip(
            estimated_close,
            min_r_close,
            max_r
        )
    )

    # Estimate the opening radius.
    #
    # The equivalent radius of small components
    # is multiplied by p_open.
    estimated_open = round(
        p_open * r_small
    )

    # Restrict the opening radius to the interval:
    #
    # min_r_open <= r_open <= max_r
    r_open = int(
        np.clip(
            estimated_open,
            min_r_open,
            max_r
        )
    )

    # Estimate the erosion radius.
    #
    # The typical internal distance is multiplied
    # by the smaller erosion multiplier p_erode.
    estimated_erode = round(
        p_erode * r_in_med
    )

    # Restrict the erosion radius to the interval:
    #
    # 0 <= r_erode <= max_r
    #
    # Unlike opening and closing, erosion may be disabled
    # by using a radius of zero.
    r_erode = int(
        np.clip(
            estimated_erode,
            0,
            max_r
        )
    )

    # Return all three morphology radii.
    #
    # max(..., 0) guarantees that no negative value
    # can be returned.
    return (
        max(r_open, 0),
        max(r_close, 0),
        max(r_erode, 0)
    )

def apply_morphology_auto(
    neuron_mask,                  # Original neuron boolean mask.
    vx_um,                        # Voxel size along X, in micrometers.
    vy_um,                        # Voxel size along Y, in micrometers.
    vz_um,                        # Voxel size along Z, in micrometers.
    ERODE_MULT=1.0,               # Extra erosion amount added in voxels.
    mode="dt",                    # Intended operation mode.
                                  # Currently not used inside this function.
    ch2_for_scoring=None,         # Optional image channel for scoring.
                                  # Currently not used.
    area_stability=(0.85, 1.15),  # Intended acceptable area-change range.
                                  # Currently not used.
):
    """
    Automatically refine a neuron mask using morphology.

    Processing order
    ----------------
    1. Estimate morphology radii.
    2. Add ERODE_MULT to the estimated erosion radius.
    3. Apply binary opening.
    4. Apply binary closing.
    5. Apply binary erosion.
    6. Remove small connected objects.

    Returns
    -------
    refined : ndarray
        Refined boolean neuron mask.

    parameters : dict
        Dictionary containing the applied opening,
        closing, and erosion radii.
    """

    # Automatically calculate the morphology radii.
    r_open, r_close, r_erode = auto_morphology_params(
        neuron_mask,
        vx_um,
        vy_um,
        vz_um
    )

    # Set the maximum permitted radius
    # for morphology operations.
    max_r = 12

    # ----------------------------------------------------------
    # Adjust the erosion radius using ERODE_MULT.
    # ----------------------------------------------------------

    # Important:
    #
    # Despite the name ERODE_MULT, this value is added
    # to the automatically estimated erosion radius.
    #
    # It is not multiplied.
    #
    # Example:
    # r_erode = 2
    # ERODE_MULT = 1.0
    #
    # adjusted radius = 2 + 1 = 3
    adjusted_erode = (
        r_erode
        + float(ERODE_MULT)
    )

    # Round the adjusted value to the nearest integer.
    adjusted_erode = int(
        round(
            adjusted_erode
        )
    )

    # Restrict the final erosion radius
    # to the valid interval from 0 to max_r.
    r_erode = int(
        np.clip(
            adjusted_erode,
            0,
            max_r
        )
    )

    def _refine(
        mask,
        ro,
        rc,
        re_
    ):
        """
        Apply morphology operations to one binary mask.

        Parameters
        ----------
        mask : ndarray
            Input boolean mask.

        ro : int
            Opening radius.

        rc : int
            Closing radius.

        re_ : int
            Erosion radius.

        Returns
        -------
        ndarray
            Refined boolean mask.
        """

        # Make a copy of the input mask.
        #
        # This prevents modification of the original array.
        out = mask.copy()

        # Check whether opening is enabled.
        if ro > 0:

            # Create a three-dimensional spherical
            # structuring element with radius ro.
            opening_structure = ball(
                ro
            )

            # Apply binary opening.
            #
            # Opening consists of:
            #   1. erosion
            #   2. dilation
            #
            # It can remove:
            #   - small isolated objects
            #   - narrow protrusions
            #   - thin foreground bridges
            out = binary_opening(
                out,
                opening_structure
            )

        # Check whether closing is enabled.
        if rc > 0:

            # Create a three-dimensional spherical
            # structuring element with radius rc.
            closing_structure = ball(
                rc
            )

            # Apply binary closing.
            #
            # Closing consists of:
            #   1. dilation
            #   2. erosion
            #
            # It can:
            #   - close small holes
            #   - fill narrow gaps
            #   - connect nearby foreground regions
            out = binary_closing(
                out,
                closing_structure
            )

        # Check whether erosion is enabled.
        if re_ > 0:

            # Create a three-dimensional spherical
            # structuring element with radius re_.
            erosion_structure = ball(
                re_
            )

            # Apply binary erosion.
            #
            # Erosion removes foreground voxels
            # from object boundaries.
            #
            # It can:
            #   - shrink objects
            #   - remove narrow extensions
            #   - separate objects joined by thin bridges
            out = binary_erosion(
                out,
                erosion_structure
            )

        # Count the number of foreground voxels
        # after the morphology operations.
        foreground_count = np.sum(
            out
        )

        # Calculate an adaptive minimum connected-object size.
        #
        # The threshold equals 0.001% of the current
        # foreground voxel count.
        adaptive_min_size = int(
            foreground_count * 1e-5
        )

        # Require a minimum threshold of at least 8 voxels.
        min_size = max(
            8,
            adaptive_min_size
        )

        # Remove connected foreground objects smaller
        # than the selected minimum size.
        #
        # connectivity=3 means full 3D connectivity:
        #   - face neighbors
        #   - edge neighbors
        #   - corner neighbors
        out = remove_small_objects(
            out,
            min_size=min_size,
            connectivity=3
        )

        # Return the refined binary mask.
        return out

    # Apply the morphology refinement using
    # the automatically selected radii.
    refined = _refine(
        neuron_mask,
        r_open,
        r_close,
        r_erode
    )

    # Return:
    #
    # 1. The refined neuron mask.
    # 2. A dictionary describing the radii that were used.
    return refined, {
        "open": r_open,
        "close": r_close,
        "erode": r_erode
    }

# ==========================================================
# FULL-SIZE EXPORT HELPERS
#
# These helper functions:
#   1. Normalize 3D image stacks to 8-bit grayscale.
#   2. Generate colors for labeled cells.
#   3. Create full-resolution RGB overlay TIFF and video files.
# ==========================================================
def _norm_u8_stack(vol):
    """
    Normalize an image volume to the unsigned 8-bit range 0–255.

    Parameters
    ----------
    vol : ndarray
        Input image volume, usually with shape (Z, Y, X).

    Returns
    -------
    ndarray
        Normalized uint8 image volume.
    """

    # Find the minimum intensity in the entire image volume.
    vmin = float(
        vol.min()
    )

    # Find the maximum intensity in the entire image volume.
    vmax = float(
        vol.max()
    )

    # Check whether the image contains a nonzero intensity range.
    if vmax > vmin:

        # Subtract the minimum so the lowest value becomes zero.
        normalized = (
            vol - vmin
        ) / (
            vmax - vmin
        )

        # Limit all normalized values to the interval 0–1.
        normalized = np.clip(
            normalized,
            0,
            1
        )

        # Scale normalized values from 0–1 to 0–255.
        normalized = normalized * 255.0

        # Convert the result to unsigned 8-bit integers.
        out = normalized.astype(
            np.uint8
        )

    else:

        # If all pixels have the same intensity,
        # create an all-zero uint8 image with the same shape.
        out = np.zeros_like(
            vol,
            dtype=np.uint8
        )

    # Return the normalized 8-bit image stack.
    return out

def make_label_colormap(
    n_labels,
    seed_hue=0.0
):
    """
    Generate one RGB color for every integer label.

    Parameters
    ----------
    n_labels : int
        Largest label number.

    seed_hue : float
        Starting hue in the HSV color space.
        Expected range is normally 0–1.

    Returns
    -------
    ndarray
        Array with shape (n_labels + 1, 3).

        Row 0 is black for the background.
        Rows 1 through n_labels contain RGB colors.
    """

    # Create an RGB color table filled with zeros.
    #
    # The table contains one extra row because label 0
    # represents the background.
    colors = np.zeros(
        (n_labels + 1, 3),
        dtype=np.uint8
    )

    # If there are no foreground labels,
    # return the background-only color table.
    if n_labels <= 0:
        return colors

    # Generate one color for every foreground label.
    for i in range(
        1,
        n_labels + 1
    ):

        # Spread hues evenly across the HSV color wheel.
        #
        # seed_hue shifts the starting color.
        # The modulo operation keeps the hue between 0 and 1.
        h = (
            seed_hue
            + (i - 1) / max(n_labels, 1)
        ) % 1.0

        # Use full saturation for vivid colors.
        s = 1.0

        # Use full brightness.
        v = 1.0

        # Convert the HSV color into floating-point RGB values.
        #
        # Each returned RGB component lies between 0 and 1.
        r, g, b = colorsys.hsv_to_rgb(
            h,
            s,
            v
        )

        # Convert RGB values from 0–1 to 0–255
        # and store them in the color table.
        colors[i] = (
            int(255 * r),
            int(255 * g),
            int(255 * b)
        )

    # Return the complete label-to-color table.
    return colors

def export_fullsize_overlay_stack(
    img_ch1,                       # First 3D image channel.
    img_ch2_raw,                   # Raw second 3D image channel.
    cell_seg_viz,                  # 3D integer cell-label image.
    df,                            # DataFrame containing lysosome measurements.
    vx_um,                         # Physical voxel size along X, in µm.
    vy_um,                         # Physical voxel size along Y, in µm.
    vz_um,                         # Physical voxel size along Z, in µm.
    output_dir,                    # Folder where outputs will be saved.
    alpha_labels=0.45,             # Opacity of colored cell labels.
    draw_only_inside=True,         # Draw only lysosomes classified inside cells.
    fps=8,                         # Frame rate for MP4 or GIF output.
    basename="FULLSIZE_overlay_CellID_Lysosomes",
):
    """
    Create a full-resolution RGB overlay stack.

    The exported overlay combines:

    - channel 1,
    - channel 2,
    - colored cell labels,
    - projected lysosome circles.

    The function saves:

    - an RGB TIFF stack,
    - an MP4 video when FFMPEG is available,
    - otherwise a GIF fallback.

    Returns
    -------
    tuple
        Paths to the TIFF and intended MP4 output files.
    """

    # Create the output directory if it does not already exist.
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Convert channel 1 to normalized uint8 intensity values.
    ch1_u8 = _norm_u8_stack(
        img_ch1.astype(np.float32)
    )

    # Convert raw channel 2 to normalized uint8 intensity values.
    ch2_u8 = _norm_u8_stack(
        img_ch2_raw.astype(np.float32)
    )

    # Read the image-stack dimensions.
    #
    # Z = number of slices.
    # H = image height.
    # W = image width.
    Z, H, W = ch2_u8.shape

    # Determine the largest cell-label value.
    #
    # If cell_seg_viz is not a NumPy array,
    # assume there are no labels.
    n_labels = (
        int(cell_seg_viz.max())
        if isinstance(cell_seg_viz, np.ndarray)
        else 0
    )

    # Generate one display color for every cell label.
    cmap = make_label_colormap(
        n_labels,
        seed_hue=0.13
    )

    # Initialize the filtered lysosome table as unavailable.
    use_df = None

    # Check that df is a nonempty DataFrame and contains
    # all required position and radius columns.
    if (
        isinstance(df, pd.DataFrame)
        and len(df) > 0
        and {
            "z_um",
            "y_um",
            "x_um",
            "radius_um"
        }.issubset(df.columns)
    ):

        # If only intracellular lysosomes should be drawn
        # and the classification column exists,
        # keep only rows classified as "cell".
        if (
            draw_only_inside
            and "location_ch2" in df.columns
        ):

            # Filter lysosomes classified as inside a cell.
            use_df = df[
                df["location_ch2"] == "cell"
            ].copy()

        else:

            # Otherwise use every lysosome row.
            use_df = df.copy()

        # Keep only rows with finite Z positions.
        valid_z = np.isfinite(
            use_df["z_um"]
        )

        # Keep only rows with finite Y positions.
        valid_y = np.isfinite(
            use_df["y_um"]
        )

        # Keep only rows with finite X positions.
        valid_x = np.isfinite(
            use_df["x_um"]
        )

        # Keep only rows with finite radius values.
        valid_radius = np.isfinite(
            use_df["radius_um"]
        )

        # Apply all validity conditions.
        use_df = use_df[
            valid_z
            & valid_y
            & valid_x
            & valid_radius
        ].copy()

    # Calculate a representative physical XY pixel size.
    #
    # The geometric mean is used in case X and Y
    # pixel dimensions are not identical.
    px_um_xy = float(
        np.sqrt(vx_um * vy_um)
    )

    # Allocate the final RGB frame stack.
    #
    # Shape:
    #   Z slices × H rows × W columns × 3 RGB channels.
    frames = np.zeros(
        (Z, H, W, 3),
        dtype=np.uint8
    )

    # Process every Z slice.
    for z in range(Z):

        # Create a three-channel base image.
        #
        # Red channel   = channel 1
        # Green channel = channel 2
        # Blue channel  = channel 1
        #
        # This produces a magenta/green-style overlay.
        base = np.dstack(
            [
                ch1_u8[z],
                ch2_u8[z],
                ch1_u8[z]
            ]
        ).astype(np.float32)

        # Extract the cell-label image for the current slice.
        lab2d = cell_seg_viz[
            z
        ].astype(np.int32)

        # Convert every integer cell label into an RGB color.
        lab_rgb = cmap[
            lab2d
        ]

        # Convert label colors to float for alpha blending.
        lab_rgb_f = lab_rgb.astype(
            np.float32
        )

        # Create a mask marking foreground cell labels.
        #
        # The additional final dimension allows broadcasting
        # over the three RGB channels.
        mask = (
            lab2d > 0
        )[..., None].astype(np.float32)

        # Blend the original image and colored cell labels.
        #
        # Outside cells:
        #   mask = 0, so only base is used.
        #
        # Inside cells:
        #   the base image and label color are mixed
        #   according to alpha_labels.
        out = (
            base * (
                1.0 - alpha_labels * mask
            )
            + lab_rgb_f * (
                alpha_labels * mask
            )
        )

        # Check whether valid lysosome measurements exist.
        if (
            use_df is not None
            and len(use_df) > 0
        ):

            # Convert lysosome Z positions from micrometers
            # into floating-point slice coordinates.
            zc = (
                use_df["z_um"].to_numpy()
                / vz_um
            ).astype(float)

            # Convert lysosome Y positions from micrometers
            # into floating-point pixel coordinates.
            yc = (
                use_df["y_um"].to_numpy()
                / vy_um
            ).astype(float)

            # Convert lysosome X positions from micrometers
            # into floating-point pixel coordinates.
            xc = (
                use_df["x_um"].to_numpy()
                / vx_um
            ).astype(float)

            # Extract lysosome radii in micrometers.
            r_um = use_df[
                "radius_um"
            ].to_numpy().astype(float)

            # Calculate the physical Z distance between
            # every lysosome center and the current slice.
            dz_um = np.abs(
                zc - z
            ) * vz_um

            # Identify lysosomes whose 3D spheres intersect
            # the current Z slice.
            hits = (
                dz_um <= r_um
            )

            # Continue only if at least one lysosome intersects.
            if np.any(hits):

                # Calculate the projected circular radius
                # on the current slice.
                #
                # For a sphere:
                #
                # projected_radius² = sphere_radius² - z_distance²
                r_proj_um = np.sqrt(
                    np.clip(
                        r_um[hits] ** 2
                        - dz_um[hits] ** 2,
                        0.0,
                        None
                    )
                )

                # Convert projected radius from micrometers
                # into equivalent XY pixels.
                r_proj_px = (
                    r_proj_um
                    / max(px_um_xy, 1e-12)
                )

                # Round lysosome Y positions to integer pixels.
                ys = np.rint(
                    yc[hits]
                ).astype(int)

                # Round lysosome X positions to integer pixels.
                xs = np.rint(
                    xc[hits]
                ).astype(int)

                # Convert the blended image into uint8
                # before drawing with OpenCV.
                out_u8 = np.clip(
                    out,
                    0,
                    255
                ).astype(np.uint8)

                # Draw one projected circle for every
                # lysosome intersecting the current slice.
                for y, x, rp in zip(
                    ys,
                    xs,
                    r_proj_px
                ):

                    # Round the projected radius to an integer.
                    #
                    # A minimum radius of 3 pixels is enforced
                    # so very small lysosomes remain visible.
                    rr = int(
                        max(
                            3,
                            round(rp)
                        )
                    )

                    # Verify that the center lies inside the image
                    # and that the radius is positive.
                    if (
                        0 <= y < H
                        and 0 <= x < W
                        and rr > 0
                    ):

                        # Draw a thick black outer circle.
                        #
                        # OpenCV coordinates are supplied as (x, y).
                        cv2.circle(
                            out_u8,
                            (x, y),
                            rr,
                            (0, 0, 0),
                            4,
                            lineType=cv2.LINE_AA
                        )

                        # Draw a thinner yellow/cyan-style inner outline.
                        #
                        # OpenCV uses BGR color order.
                        # (0, 255, 255) is yellow in BGR.
                        cv2.circle(
                            out_u8,
                            (x, y),
                            rr,
                            (0, 255, 255),
                            2,
                            lineType=cv2.LINE_AA
                        )

                # Convert the annotated image back to float
                # so it remains compatible with later processing.
                out = out_u8.astype(
                    np.float32
                )

        # Store the completed RGB slice in the frame stack.
        frames[z] = np.clip(
            out,
            0,
            255
        ).astype(np.uint8)

    # Build the output path for the RGB TIFF stack.
    tiff_path = os.path.join(
        output_dir,
        f"{basename}.tif"
    )

    # Save all RGB slices as a TIFF stack.
    tiff.imwrite(
        tiff_path,
        frames,
        photometric="rgb"
    )

    # Print the saved TIFF path.
    print(
        "Saved full-size RGB TIFF stack:",
        tiff_path
    )

    # Build the intended MP4 output path.
    mp4_path = os.path.join(
        output_dir,
        f"{basename}.mp4"
    )

    try:

        # Open an FFMPEG video writer.
        with imageio.get_writer(
            mp4_path,
            fps=int(fps),
            format="FFMPEG",
            codec="libx264",
            macro_block_size=None
        ) as w:

            # Add every RGB slice as one video frame.
            for fr in frames:
                w.append_data(
                    fr
                )

        # Print the saved MP4 path.
        print(
            "Saved full-size MP4:",
            mp4_path
        )

    except Exception as e:

        # If MP4 generation fails, create a GIF instead.
        gif_path = os.path.join(
            output_dir,
            f"{basename}.gif"
        )

        # Save the RGB frames as an animated GIF.
        imageio.mimsave(
            gif_path,
            list(frames),
            fps=int(fps)
        )

        # Report the GIF path and original FFMPEG error.
        print(
            "FFMPEG failed, saved GIF instead:",
            gif_path,
            "Error:",
            e
        )

    # Return the TIFF path and the intended MP4 path.
    #
    # Important:
    # if FFMPEG failed, mp4_path may not refer to an existing file.
    return tiff_path, mp4_path

# ==========================================================
# HELPER: ASK FOR VOXEL SIZE ONLY WHEN METADATA IS MISSING
# ==========================================================
def _ask_missing_scale(
    param_label,    # Human-readable parameter name.
    default_val,    # Default numerical value.
    unit_text       # Unit displayed in the dialog.
):
    """
    Ask the user to accept a default scale or enter a new one.

    A temporary hidden Tkinter root window is created because
    message boxes and simple dialogs require a Tkinter parent.
    """

    # Create a temporary Tkinter window.
    rr = tk.Tk()

    # Hide the temporary root window.
    rr.withdraw()

    try:

        # Ask whether the user wants to use the default value.
        use_def = messagebox.askyesno(
            "Missing metadata",
            f"{param_label} is missing in metadata.\n\n"
            f"Use default: {default_val} {unit_text}?\n\n"
            f"Yes = use default\nNo = enter new value"
        )

        # Start with the supplied default value.
        val = float(
            default_val
        )

        # If the user selects No,
        # ask for a manually entered value.
        if not use_def:

            # Open a floating-point input dialog.
            v = simpledialog.askfloat(
                "Enter value",
                f"Enter {param_label} ({unit_text}):",
                initialvalue=float(default_val),
                minvalue=0.0
            )

            # If the user entered a value rather than cancelling,
            # replace the default value.
            if v is not None:
                val = float(
                    v
                )

        # Return the chosen value as a Python float.
        return float(
            val
        )

    finally:

        # Always destroy the temporary Tkinter window,
        # even if an error occurs or the dialog is cancelled.
        rr.destroy()

# ==========================================================
# NAPARI EDIT-BLOCK ADAPTER SETTINGS
# ==========================================================
# Disable neurite-specific operating mode.
NEURITE_MODE = False

# Enable lysosome-table editing in Napari.
EDIT_LYSOSOME_TABLE_IN_NAPARI = True

# Filename used for the edited lysosome CSV table.
LYSOSOME_EDITED_CSV = (
    "lysosomes_with_cell_vs_outside_EDITED.csv"
)

def attach_all_blob_fields(
    df_in: pd.DataFrame
) -> pd.DataFrame:
    """
    Ensure that a lysosome DataFrame contains all required columns.

    Missing columns are created with safe default values.
    Existing numerical identifier columns are converted to integers.
    """

    # Copy the input DataFrame so the original is not modified.
    df = df_in.copy()

    # Ensure that all three physical position columns exist.
    for col in (
        "z_um",
        "y_um",
        "x_um"
    ):

        # If the coordinate column is missing,
        # create it and fill it with NaN values.
        if col not in df.columns:
            df[col] = np.nan

    # If radius is missing but diameter exists,
    # calculate radius as half of diameter.
    if (
        "radius_um" not in df.columns
        and "diameter_um" in df.columns
    ):

        # Convert diameter values to numeric.
        #
        # Invalid values become NaN.
        numeric_diameter = pd.to_numeric(
            df["diameter_um"],
            errors="coerce"
        )

        # Calculate radius from diameter.
        df["radius_um"] = (
            numeric_diameter / 2.0
        )

    # If diameter is missing but radius exists,
    # calculate diameter as twice the radius.
    if (
        "diameter_um" not in df.columns
        and "radius_um" in df.columns
    ):

        # Convert radius values to numeric.
        numeric_radius = pd.to_numeric(
            df["radius_um"],
            errors="coerce"
        )

        # Calculate diameter from radius.
        df["diameter_um"] = (
            numeric_radius * 2.0
        )

    # Add the location-classification column if missing.
    if "location_ch2" not in df.columns:

        # Default every lysosome to outside a cell.
        df["location_ch2"] = "outside"

    # Convert location values to strings.
    df["location_ch2"] = df[
        "location_ch2"
    ].astype(str)

    # Add the cell-ID column if missing.
    if "cell_id_ch2" not in df.columns:

        # Use zero to represent no assigned cell.
        df["cell_id_ch2"] = 0

    # Convert cell IDs to numeric values.
    #
    # Invalid entries become NaN, then zero,
    # and are finally converted to integers.
    df["cell_id_ch2"] = pd.to_numeric(
        df["cell_id_ch2"],
        errors="coerce"
    ).fillna(
        0
    ).astype(
        int
    )

    # Add the peak-intensity column if missing.
    if "peak_gray" not in df.columns:

        # Use NaN because no peak intensity is available.
        df["peak_gray"] = np.nan

    # Process optional serial-ID columns.
    for col in (
        "cell_id_serial",
        "lys_id_serial"
    ):

        # Convert the column only if it exists.
        if col in df.columns:

            # Invalid entries become zero.
            # The final type is integer.
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(
                0
            ).astype(
                int
            )

    # Return the completed DataFrame.
    return df

# ==========================================================
# MAIN PROGRAM
# ==========================================================
# Default physical XY pixel size in micrometers.
DEFAULT_VX_VY_UM = 0.04

# Default Z spacing.
#
# None means that no fixed default is provided here.
DEFAULT_VZ_UM = None

# Open the GUI and collect all processing parameters.
cfg = get_user_config_gui(
    default_vxy_um=DEFAULT_VX_VY_UM,
    default_vz_um=DEFAULT_VZ_UM,
    default_erode_mult=1.0,
    default_blob_threshold=0.001,
)


# Read the selected input-file path from the configuration.
file_path = cfg[
    "file_path"
]

# Read the selected output-directory path.
output_dir = cfg[
    "output_dir"
]

# Create the output directory if necessary.
os.makedirs(
    output_dir,
    exist_ok=True
)


def outpath(name):
    """
    Build a full output path inside output_dir.
    """

    # Join the output folder and filename.
    return os.path.join(
        output_dir,
        name
    )

# Display the selected input file.
print(
    "Selected file:",
    file_path
)

# Display the output directory.
print(
    "Outputs will be saved to:",
    output_dir
)


# Read the erosion adjustment setting.
ERODE_MULT = cfg[
    "ERODE_MULT"
]

# Read the Laplacian-of-Gaussian blob-detection threshold.
BLOB_THRESHOLD = cfg[
    "BLOB_THRESHOLD"
]


# Read the optional minimum lysosome diameter.
DIAMETER_MIN_UM = cfg[
    "DIAMETER_MIN_UM"
]

# Read the optional maximum lysosome diameter.
DIAMETER_MAX_UM = cfg[
    "DIAMETER_MAX_UM"
]


# Read the largest physically reasonable XY pixel size.
MAX_REASONABLE_VXY_UM = cfg[
    "MAX_REASONABLE_VXY_UM"
]

# Read the distance margin used around masks.
MARGIN_UM = cfg[
    "MARGIN_UM"
]

# Read the required overlap fraction.
OVERLAP_ALPHA = cfg[
    "OVERLAP_ALPHA"
]

# Read the maximum neighbor distance in voxels.
NEIGHBOR_MAX_VOX = cfg[
    "NEIGHBOR_MAX_VOX"
]

# Read the minimum visualization object size.
VIZ_MIN_VOXELS = cfg[
    "VIZ_MIN_VOXELS"
]

# Read the distance-map smoothing sigma.
DIST_SMOOTH_SIGMA = cfg[
    "DIST_SMOOTH_SIGMA"
]

# Read the H-maxima suppression value.
H_MAXIMA = cfg[
    "H_MAXIMA"
]


# Read the minimum number of adaptive membrane bins.
MEMBRANE_MINIMUM_BINS = cfg[
    "MEMBRANE_MINIMUM_BINS"
]

# Read the maximum number of adaptive membrane bins.
MEMBRANE_MAXIMUM_BINS = cfg[
    "MEMBRANE_MAXIMUM_BINS"
]

# Read the membrane-profile smoothing sigma.
MEMBRANE_SIGMA_SMOOTH = cfg[
    "MEMBRANE_SIGMA_SMOOTH"
]

# Read the relative peak-height parameter.
MEMBRANE_REL_HEIGHT = cfg[
    "MEMBRANE_REL_HEIGHT"
]


# Read the smoothing sigma applied to channel 1.
CH1_SMOOTH_SIGMA = cfg[
    "CH1_SMOOTH_SIGMA"
]

# Read the minimum blob-detector sigma.
BLOB_MIN_SIGMA = cfg[
    "BLOB_MIN_SIGMA"
]

# Read the maximum blob-detector sigma.
BLOB_MAX_SIGMA = cfg[
    "BLOB_MAX_SIGMA"
]

# Read the number of tested blob sigma values.
BLOB_NUM_SIGMA = cfg[
    "BLOB_NUM_SIGMA"
]


# Read the maximum radial-profile radius in nanometers.
RADIAL_MAX_RADIUS_NM = cfg[
    "RADIAL_MAX_RADIUS_NM"
]

# Read the radial-profile sampling interval in nanometers.
RADIAL_DR_NM = cfg[
    "RADIAL_DR_NM"
]

# Read the minimum radial intensity-drop fraction.
RADIAL_MIN_DROP_FRACTION = cfg[
    "RADIAL_MIN_DROP_FRACTION"
]


# Read the smoothing sigma applied to channel 2.
CH2_SMOOTH_SIGMA = cfg[
    "CH2_SMOOTH_SIGMA"
]

# Read the adaptive-threshold block size.
THRESH_BLOCK_SIZE = cfg[
    "THRESH_BLOCK_SIZE"
]

# Read the threshold-offset standard-deviation multiplier.
THRESH_OFFSET_STD_MULT = cfg[
    "THRESH_OFFSET_STD_MULT"
]


# Read the output-video frame rate.
FPS = cfg[
    "VIDEO_FPS"
]

# Read whether Napari should launch automatically.
LAUNCH_VIEWER = cfg[
    "LAUNCH_VIEWER"
]

# Read whether videos should be generated.
GENERATE_VIDEOS = cfg[
    "GENERATE_VIDEOS"
]

# ----------------------------------------------------------
# Load image data and metadata.
# ----------------------------------------------------------

# Load both image channels, physical voxel sizes,
# and basic file metadata.
img_ch1, img_ch2, (
    vx_um,
    vy_um,
    vz_um
), meta = load_any(
    file_path
)

# Display voxel sizes obtained from image metadata.
print(
    f"[metadata] vx_um={vx_um}  "
    f"vy_um={vy_um}  "
    f"vz_um={vz_um} "
)


# Check whether either XY pixel dimension is missing.
if vx_um is None or vy_um is None:

    # Ask the user whether to use the default XY pixel size
    # or enter a different value.
    vxy = _ask_missing_scale(
        "XY pixel size",
        cfg["DEFAULT_VX_VY_UM"],
        "µm/px"
    )

    # Use the same value for both X and Y dimensions.
    vx_um = float(
        vxy
    )

    # Assign the chosen value to Y as well.
    vy_um = float(
        vxy
    )


# Check whether Z spacing is missing.
if vz_um is None:

    # Use the configured default Z size when one exists.
    if cfg["DEFAULT_VZ_UM"] is not None:

        # Convert the configured default to float.
        default_z = float(
            cfg["DEFAULT_VZ_UM"]
        )

    else:

        # If no Z default exists, use the X pixel size.
        default_z = float(
            vx_um
        )

    # Ask the user to accept or replace the proposed Z spacing.
    vz_um = _ask_missing_scale(
        "Z step size",
        default_z,
        "µm/slice"
    )


# Validate the XY pixel size.
if vx_um > MAX_REASONABLE_VXY_UM:

    # Stop processing if the XY pixel size appears unreasonable.
    raise ValueError(
        f"XY pixel size too large: "
        f"{vx_um} µm/px"
    )


# Calculate one representative XY pixel dimension.
#
# The geometric mean supports slightly anisotropic XY pixels.
px_um_xy = float(
    np.sqrt(vx_um * vy_um)
)

# Create a second variable containing the same XY pixel size.
#
# Multiplication by 1 does not change the value.
px_um = px_um_xy * 1

# Calculate the physical volume of one voxel.
#
# Units:
#   µm × µm × µm = µm³.
voxel_um3 = (
    vx_um
    * vy_um
    * vz_um
)

# Display the final voxel dimensions used for processing.
print(
    f"Voxel size (µm): "
    f"X={vx_um}, "
    f"Y={vy_um}, "
    f"Z={vz_um}"
)

# ==========================================================
# IMAGE ALIASES
# ==========================================================
# Assign channel 1 to a shorter variable name.
# Channel 1 contains the lysosome signal.
image = img_ch1

# Assign channel 2 to a shorter variable name.
# Channel 2 contains the cell signal.
image_2 = img_ch2

# ==========================================================
# LYSOSOME DETECTION IN CHANNEL 1
# ==========================================================
# Smooth the lysosome image using a Gaussian filter.
#
# This reduces small intensity fluctuations and noise before
# applying the Laplacian-of-Gaussian blob detector.
image_smooth = gaussian(
    image,
    sigma=CH1_SMOOTH_SIGMA
)

# Detect bright approximately spherical structures
# in the smoothed channel-1 image.
blobs = blob_log(
    image_smooth,

    # Smallest Gaussian sigma tested by blob_log.
    min_sigma=BLOB_MIN_SIGMA,

    # Largest Gaussian sigma tested by blob_log.
    max_sigma=BLOB_MAX_SIGMA,

    # Number of sigma values tested between the minimum
    # and maximum sigma values.
    num_sigma=BLOB_NUM_SIGMA,

    # Minimum detector-response threshold.
    #
    # Lower values generally detect more candidate blobs.
    threshold=BLOB_THRESHOLD
)


# Check whether blob_log returned None.
if blobs is None:

    # Replace None with an empty array having four columns.
    #
    # The expected blob format is:
    # [z_position, y_position, x_position, sigma_or_radius]
    blobs = np.zeros(
        (0, 4),
        dtype=float
    )

# Continue with radius refinement only when at least one
# blob was detected.
if len(blobs) > 0:

    # Convert the blob array to 32-bit floating point.
    #
    # This permits later radius values to contain decimals.
    blobs = blobs.astype(
        np.float32
    )

    # Ignore the original sigma returned by blob_log.
    #
    # Column 3 will be replaced by the radius estimated
    # by the refinement functions.
    blobs[:, 3] = 0

    # Refine blob sizes using a local 3D Gaussian fit.
    blobs = refine_radii_via_3d_gaussian_fit(
        image_smooth,   # Smoothed lysosome image.
        blobs,          # Blob center coordinates.
        vx_um,          # X voxel size in micrometers.
        vy_um,          # Y voxel size in micrometers.
        vz_um,          # Z voxel size in micrometers.
        win_um=0.5      # Half-width of the fitting region in micrometers.
    )

    # Refine blob sizes again using the radial intensity profile.
    blobs = refine_radii_via_radial_intensity(
        image_smooth,                         # Smoothed lysosome image.
        blobs,                                # Gaussian-refined blobs.
        vx_um,                                # X voxel size in micrometers.
        vy_um,                                # Y voxel size in micrometers.
        vz_um,                                # Z voxel size in micrometers.
        max_radius_nm=RADIAL_MAX_RADIUS_NM,   # Maximum radial distance in nm.
        dr_nm=RADIAL_DR_NM,                   # Radial-bin spacing in nm.
        min_drop_fraction=RADIAL_MIN_DROP_FRACTION
                                               # Minimum intensity-drop parameter.
    )

# ==========================================================
# PEAK INTENSITY MEASUREMENT IN RAW CHANNEL 1
# ==========================================================
# Allocate an array for the local peak intensity of each blob.
#
# uint16 is appropriate when the original image is 16-bit.
peak_gray = np.zeros(
    len(blobs),
    dtype=np.uint16
)

# Read the dimensions of the raw channel-1 image.
#
# Z0 = number of slices.
# Y0 = image height.
# X0 = image width.
Z0, Y0, X0 = image.shape

# Define a one-voxel search radius around each blob center.
#
# Away from image edges, this creates a 3 × 3 × 3 patch.
rad = 1

# Allocate an array for the Z slice containing the local
# maximum intensity around each blob.
peak_slice = np.zeros(
    len(blobs),
    dtype=int
)

# Process every detected blob.
for i, (zc, yc, xc, _) in enumerate(blobs):

    # Round the blob's Z coordinate to the nearest slice.
    zc_i = int(
        round(zc)
    )

    # Round the blob's Y coordinate to the nearest pixel.
    yc_i = int(
        round(yc)
    )

    # Round the blob's X coordinate to the nearest pixel.
    xc_i = int(
        round(xc)
    )

    # Calculate the lower Z boundary of the local patch.
    #
    # max(0, ...) prevents a negative array index.
    z1 = max(
        0,
        zc_i - rad
    )

    # Calculate the upper exclusive Z boundary.
    #
    # min(Z0, ...) prevents indexing past the final slice.
    z2 = min(
        Z0,
        zc_i + rad + 1
    )

    # Calculate the lower Y boundary.
    y1 = max(
        0,
        yc_i - rad
    )

    # Calculate the upper exclusive Y boundary.
    y2 = min(
        Y0,
        yc_i + rad + 1
    )

    # Calculate the lower X boundary.
    x1 = max(
        0,
        xc_i - rad
    )

    # Calculate the upper exclusive X boundary.
    x2 = min(
        X0,
        xc_i + rad + 1
    )

    # Extract a small local patch from the raw channel-1 image.
    patch = image[
        z1:z2,
        y1:y2,
        x1:x2
    ]

    # Find the position of the maximum intensity in the patch.
    #
    # np.argmax returns a flattened index.
    # np.unravel_index converts it back into:
    # (local_z, local_y, local_x).
    idx = np.unravel_index(
        np.argmax(patch),
        patch.shape
    )

    # Store the maximum raw intensity for this blob.
    peak_gray[i] = patch[
        idx
    ]

    # Convert the local Z index of the maximum into
    # the corresponding global image-slice index.
    peak_slice[i] = (
        z1 + idx[0]
    )


# Replace the peak-intensity slice values with the rounded
# Z coordinate of each blob center.
#
# Important:
# This line discards the peak-slice values calculated
# inside the loop above.
peak_slice = np.round(
    blobs[:, 0]
).astype(int)

# ==========================================================
# CONVERT BLOB MEASUREMENTS INTO PHYSICAL UNITS
# ==========================================================
# Continue only when at least one blob exists.
if len(blobs) > 0:

    # Convert Z coordinates from image slices to micrometers.
    z_um = (
        blobs[:, 0] * vz_um
    )

    # Convert Y coordinates from pixels to micrometers.
    y_um = (
        blobs[:, 1] * vy_um
    )

    # Convert X coordinates from pixels to micrometers.
    x_um = (
        blobs[:, 2] * vx_um
    )

    # Convert blob radius from equivalent XY pixels
    # to micrometers.
    radius_um = (
        blobs[:, 3] * px_um_xy
    )

    # Set the maximum allowed lysosome radius.
    #
    # Units are micrometers.
    MAX_RADIUS_UM = 0.4

    # Restrict every radius to the interval:
    #
    # 0.0 <= radius <= 0.4 µm
    radius_um = np.clip(
        radius_um,
        0.0,
        MAX_RADIUS_UM
    )

    # Convert the capped physical radius back to pixels.
    #
    # This updates the blobs array so all downstream code
    # uses the same capped radii.
    blobs[:, 3] = (
        radius_um / px_um_xy
    )

    # Calculate lysosome diameter.
    #
    # diameter = 2 × radius
    diameter_um = (
        2 * radius_um
    )

    # Estimate lysosome volume by assuming each lysosome
    # is a sphere.
    #
    # Sphere volume:
    # V = 4/3 × pi × radius³
    volume_um3 = (
        (4 / 3)
        * np.pi
        * radius_um**3
    )

    # Create sequential lysosome IDs beginning at 1.
    blob_ids = np.arange(
        1,
        len(blobs) + 1,
        dtype=int
    )

# Handle the case where no blobs were detected.
else:
    # Create empty arrays for all physical measurements.
    z_um = np.array([])

    y_um = np.array([])

    x_um = np.array([])

    radius_um = np.array([])

    diameter_um = np.array([])

    volume_um3 = np.array([])

    # Create an empty integer array for blob IDs.
    blob_ids = np.array(
        [],
        dtype=int
    )

# ==========================================================
# CREATE THE INITIAL LYSOSOME TABLE
# ==========================================================
# Create a pandas DataFrame containing one row per blob.
df = pd.DataFrame({

    # Sequential lysosome identifier.
    "id": blob_ids,

    # Physical Z coordinate in micrometers.
    "z_um": z_um,

    # Physical Y coordinate in micrometers.
    "y_um": y_um,

    # Physical X coordinate in micrometers.
    "x_um": x_um,

    # Radius in micrometers.
    "radius_um": radius_um,

    # Diameter in micrometers.
    "diameter_um": diameter_um,

    # Estimated spherical volume in cubic micrometers.
    "volume_um3": volume_um3,

    # Local maximum intensity measured from raw channel 1.
    "peak_gray": peak_gray,

    # Selected Z slice.
    "slice": peak_slice,
})


# Save the initial blob-measurement table as a CSV file.
df.to_csv(
    outpath(
        "lysosome_blobs_regions.csv"
    ),
    index=False
)

# Print the path of the saved file.
print(
    "Saved:",
    outpath(
        "lysosome_blobs_regions.csv"
    )
)

# ==========================================================
# HELPER FUNCTION: MAKE DUPLICATE RADII UNIQUE
# ==========================================================
def _unique_radii_within_5pct(
    radius_series,     # Pandas Series containing radius values.
    low=0.0,           # Minimum permitted radius in micrometers.
    high=0.4,          # Maximum permitted radius in micrometers.
    max_frac=0.05      # Maximum permitted relative change.
):
    """
    Make duplicate radius values numerically unique.

    The first occurrence of a radius remains unchanged.

    Later duplicate values are shifted slightly upward
    or downward while attempting to remain:

    - within max_frac of the original radius,
    - above the lower limit,
    - below the upper limit.

    Returns
    -------
    pandas.Series
        Adjusted radius values.
    """

    # Convert the input Series into a floating-point NumPy array.
    arr = radius_series.to_numpy().astype(
        float
    )

    # Allocate an output array with the same shape as arr.
    uniq = np.empty_like(
        arr,
        dtype=float
    )

    # Create a set for tracking radius values that have
    # already been assigned.
    used = set()

    # Process every radius in order.
    for i, r in enumerate(arr):

        # Restrict the original radius to the permitted range.
        base = float(
            np.clip(
                r,
                low,
                high
            )
        )

        # Calculate the largest permitted deviation.
        #
        # Normally:
        # deviation = 5% of the radius.
        #
        # A small minimum scale prevents the deviation
        # from becoming exactly zero for zero-radius values.
        dev = max(
            max_frac * max(
                abs(base),
                1e-6
            ),
            1e-9
        )

        # If this exact radius has not yet been used,
        # retain it without adjustment.
        if base not in used:

            # Store the original clipped value.
            uniq[i] = base

            # Record the value as used.
            used.add(
                base
            )

            # Continue to the next radius.
            continue

        # Record whether a suitable unused radius is found.
        found = False

        # Divide the maximum permitted deviation into
        # 20 small search steps.
        step = (
            dev / 20.0
        )

        # Search for an unused radius close to the original.
        for k in range(
            1,
            401
        ):

            # Alternate between increasing and decreasing
            # the radius.
            #
            # Odd iterations move upward.
            # Even iterations move downward.
            sgn = (
                1.0
                if k % 2 == 1
                else -1.0
            )

            # Calculate a candidate radius.
            #
            # The displacement is limited to dev.
            cand = float(
                np.clip(
                    base
                    + sgn * min(
                        dev,
                        k * step
                    ),
                    low,
                    high
                )
            )

            # Accept the candidate when:
            #
            # 1. It remains within the allowed deviation.
            # 2. It has not already been used.
            if (
                abs(cand - base) <= dev
                and cand not in used
            ):

                # Store the candidate radius.
                uniq[i] = cand

                # Mark it as used.
                used.add(
                    cand
                )

                # Record that the search succeeded.
                found = True

                # Stop searching for this radius.
                break

        # Handle the rare case where no unused candidate
        # was found during the search.
        if not found:

            # Move to the next representable floating-point value
            # in the direction of the upper limit.
            cand = float(
                np.clip(
                    np.nextafter(
                        base,
                        high
                    ),
                    low,
                    high
                )
            )

            # Store the fallback value.
            uniq[i] = cand

            # Mark the fallback value as used.
            used.add(
                cand
            )

    # Return the adjusted radii as a pandas Series.
    #
    # Preserve the original DataFrame index.
    return pd.Series(
        uniq,
        index=radius_series.index,
        name="radius_um"
    )

# ==========================================================
# CREATE A TABLE WITH UNIQUE RADIUS VALUES
# ==========================================================
# Continue only when the DataFrame contains rows.
if len(df) > 0:

    # Copy the original lysosome table.
    df_unique = df.copy()

    # Adjust duplicate radius values so they become
    # numerically unique while remaining within 5%.
    df_unique["radius_um"] = _unique_radii_within_5pct(
        df_unique["radius_um"],

        # Minimum allowed radius.
        low=0.0,

        # Maximum allowed radius.
        high=0.4,

        # Maximum relative adjustment of 5%.
        max_frac=0.05
    )

    # Recalculate diameter using the adjusted radius.
    df_unique["diameter_um"] = (
        2.0
        * df_unique["radius_um"]
    )

    # Recalculate spherical volume using the adjusted radius.
    df_unique["volume_um3"] = (
        (4.0 / 3.0)
        * np.pi
        * (
            df_unique["radius_um"] ** 3
        )
    )

    # Save the unique-radius table as a CSV file.
    df_unique.to_csv(
        outpath(
            "lysosome_blobs_regions_unique_radius.csv"
        ),
        index=False
    )

    # Print the saved file path.
    print(
        "Saved:",
        outpath(
            "lysosome_blobs_regions_unique_radius.csv"
        )
    )

    # Replace the working DataFrame with the adjusted table.
    df = df_unique.copy()


# Create a separate copy containing the current complete
# lysosome table.
df_all = df.copy()

# ==========================================================
# MEMBRANE SEGMENTATION: CELL VERSUS OUTSIDE
#
# This section:
#   1. Normalizes channel 2.
#   2. Smooths channel 2.
#   3. Creates an initial binary cell mask.
#   4. Applies automatic morphology.
#   5. Fills cell interiors.
#   6. Separates cells using marker-controlled watershed.
#   7. Detects an adaptive membrane layer for every cell.
# ==========================================================
# Convert channel 2 to 32-bit floating point.
#
# Floating-point data are needed for normalization,
# Gaussian smoothing, threshold calculations, and profiling.
vol = image_2.astype(
    np.float32
)


# Find the minimum intensity in channel 2.
vmin = float(
    vol.min()
)

# Find the maximum intensity in channel 2.
vmax = float(
    vol.max()
)

# Check whether the image contains a nonzero intensity range.
if vmax > vmin:

    # Subtract the minimum so the lowest intensity becomes zero.
    vol = (
        vol - vmin
    ) / (
        vmax - vmin
    )

    # After this operation, vol is approximately normalized
    # to the interval from 0 to 1.

else:

    # If every voxel has the same intensity,
    # set the complete volume to zero.
    vol[:] = 0.0

# Smooth the normalized channel-2 volume.
#
# Gaussian smoothing reduces noise before local thresholding.
#
# preserve_range=True prevents scikit-image from performing
# any additional intensity-range conversion.
ch2 = gaussian(
    vol,
    sigma=CH2_SMOOTH_SIGMA,
    preserve_range=True
)

# Create an empty Boolean mask with the same shape as channel 2.
#
# False means background or outside.
# True means preliminary cell or neuron signal.
neuron_mask = np.zeros_like(
    ch2,
    dtype=bool
)

# Process every Z slice independently.
for z in range(
    ch2.shape[0]
):

    # Extract the current 2D channel-2 slice.
    R = ch2[
        z
    ]

    # Calculate an adaptive local threshold image.
    #
    # Each pixel receives a threshold based on its surrounding
    # neighborhood rather than one global threshold.
    t = threshold_local(
        R,

        # Width and height of the local threshold neighborhood.
        #
        # This value should normally be an odd integer.
        block_size=THRESH_BLOCK_SIZE,

        # Shift the local threshold according to the standard
        # deviation of intensities in the current slice.
        #
        # Because the value passed to offset is negative,
        # threshold_local effectively raises the threshold
        # by THRESH_OFFSET_STD_MULT × standard deviation.
        offset=(
            -THRESH_OFFSET_STD_MULT
            * np.std(R)
        )
    )

    # Mark pixels as foreground where their intensity
    # is greater than the calculated local threshold.
    neuron_mask[z] = (
        R > t
    )

# Apply automatically estimated opening, closing, and erosion.
#
# The returned values are:
#   neuron_mask_auto = refined Boolean mask
#   chosen = dictionary of morphology radii
neuron_mask_auto, chosen = apply_morphology_auto(
    neuron_mask,
    vx_um,
    vy_um,
    vz_um,

    # User-controlled additive erosion adjustment.
    ERODE_MULT=ERODE_MULT,

    # Requested automatic distance-transform mode.
    #
    # Note: in the previously shown function, mode is accepted
    # but is not currently used internally.
    mode="dt"
)

# Display the morphology radii that were selected.
print(
    f"[auto-morphology] radii -> "
    f"open:{chosen['open']}  "
    f"close:{chosen['close']}  "
    f"erode:{chosen['erode']}"
)

# Make a copy of the automatically refined mask.
#
# This preserves neuron_mask_auto if total_mask is modified later.
total_mask = neuron_mask_auto.copy()

# Fill enclosed holes inside foreground components.
#
# This converts cell outlines or membrane-enclosed regions
# into solid filled cell regions.
filled_mask = binary_fill_holes(
    total_mask
)

# Important processing distinction:
#
# filled_mask is used to segment complete cells and assign IDs.
#
# A membrane-shell mask is intended for membrane-specific
# intensity, volume, and voxel measurements.

# Store the filled mask under a descriptive variable name.
neuron_mask_for_segmentation = filled_mask

# ==========================================================
# CELL-SEPARATION MARKERS FROM DISTANCE-MAP PEAKS
#
# The distance transform gives high values near the centers
# of thick foreground regions.
#
# H-maxima identifies significant peaks in that distance map.
# Those peaks are used as watershed seeds.
# ==========================================================
# Calculate the Euclidean distance transform inside
# the filled cell mask.
#
# Each foreground voxel receives its distance to the nearest
# background voxel.
#
# This version does not use physical voxel sampling, so the
# returned distances are expressed in voxel units.
dist_inside = edt(
    neuron_mask_for_segmentation
).astype(
    np.float32
)

# Smooth the distance map.
#
# Smoothing suppresses small irregular peaks that might
# otherwise create too many watershed regions.
dist_smooth = gaussian(
    dist_inside,
    sigma=DIST_SMOOTH_SIGMA,
    preserve_range=True
).astype(
    np.float32
)

# Detect significant regional maxima in the smoothed
# distance transform.
#
# H_MAXIMA suppresses peaks whose height is smaller
# than the selected value.
maxima = h_maxima(
    dist_smooth,
    h=H_MAXIMA
)

# Label connected maximum regions.
#
# Each connected maxima region becomes one watershed marker.
#
# connectivity=3 uses full three-dimensional connectivity.
markers = label(
    maxima,
    connectivity=3
)

# Determine how many watershed markers were created.
#
# Since marker IDs begin at 1, the maximum marker value
# equals the number of markers.
n_markers = int(
    markers.max()
)

# Print the number of detected watershed seeds.
print(
    f"[markers] seeds found: {n_markers}"
)

# Check whether at least one valid watershed marker exists.
if n_markers > 0:

    # Apply marker-controlled watershed segmentation.
    cell_seg = watershed(

        # Negate the distance map so cell centers become basins.
        -dist_smooth,

        # Use the H-maxima regions as watershed starting points.
        markers=markers,

        # Restrict watershed segmentation to the filled mask.
        mask=neuron_mask_for_segmentation
    )

else:

    # If no seeds were found, create an all-zero label image.
    #
    # Zero represents background or no detected cell.
    cell_seg = np.zeros_like(
        neuron_mask_for_segmentation,
        dtype=np.int32
    )

# Print the largest cell label.
#
# Because labels begin at 1, this normally equals
# the total number of detected cells.
print(
    f"Detected {int(cell_seg.max())} cells."
)


# Print the total number of voxels in the filled cell mask.
print(
    "filled cell voxels:",
    int(
        neuron_mask_for_segmentation.sum()
    )
)

# Create an empty Boolean mask that will contain
# the adaptively detected membrane regions.
adaptive_membrane_mask = np.zeros_like(
    cell_seg,
    dtype=bool
)

# ==========================================================
# ADAPTIVE MEMBRANE MASK
#
# For each segmented cell:
#   1. Measure distance from the cell boundary.
#   2. Group voxels into automatic distance bins.
#   3. Calculate mean channel-2 intensity in each bin.
#   4. Smooth the radial intensity profile.
#   5. Find the strongest membrane-associated peak.
#   6. Measure the peak width.
#   7. Convert that interval into a 3D membrane mask.
#
# Automatic radial binning uses the Freedman–Diaconis rule.
# Membrane thickness is estimated with peak-width analysis.
# ==========================================================
# Reinitialize the adaptive membrane mask.
#
# This second initialization replaces the identical
# initialization performed above.
adaptive_membrane_mask = np.zeros_like(
    cell_seg,
    dtype=bool
)

# Find all unique integer labels in the cell segmentation.
cell_ids = np.unique(
    cell_seg
)

# Remove label 0 because it represents the background.
cell_ids = cell_ids[
    cell_ids != 0
]

# Print the number of cells that will be analyzed.
print(
    f"[adaptive membrane] Processing "
    f"{len(cell_ids)} cells..."
)

# Process every segmented cell independently.
for cid in cell_ids:

    # Create a Boolean mask for the current cell.
    #
    # True means the voxel belongs to this cell ID.
    cell = (
        cell_seg == cid
    )

    # Count the number of voxels belonging to this cell.
    cell_voxels = int(
        cell.sum()
    )


    # Skip very small segmented cells.
    if cell_voxels < 500:

        # Explain why this cell was excluded.
        print(
            f"[adaptive membrane] Cell {cid}: "
            f"skipped because it has only "
            f"{cell_voxels} voxels"
        )

        # Move to the next cell.
        continue

    # ------------------------------------------------------
    # DISTANCE FROM THE CELL BOUNDARY
    # ------------------------------------------------------

    # Calculate the Euclidean distance transform inside
    # the current cell.
    #
    # sampling supplies the physical voxel dimensions in:
    #   Z, Y, X order.
    #
    # The resulting distances are therefore expressed
    # directly in micrometers.
    dist = distance_transform_edt(
        cell,
        sampling=(
            vz_um,
            vy_um,
            vx_um
        )
    ).astype(
        np.float32
    )

    # Extract distance values only for voxels
    # belonging to the current cell.
    #
    # Convert to float64 for statistical calculations.
    d = dist[
        cell
    ].astype(
        np.float64
    )

    # Extract raw channel-2 intensities for the same
    # voxels belonging to the current cell.
    intensity = image_2[
        cell
    ].astype(
        np.float64
    )

    # ------------------------------------------------------
    # REMOVE INVALID VALUES
    # ------------------------------------------------------

    # Identify positions where both the distance
    # and intensity are finite numbers.
    valid = (
        np.isfinite(d)
        & np.isfinite(intensity)
    )

    # Keep only valid distance measurements.
    d = d[
        valid
    ]

    # Keep only the corresponding valid intensities.
    intensity = intensity[
        valid
    ]

    # Require at least 20 valid voxels for a useful profile.
    if d.size < 20:

        # Explain why the cell was excluded.
        print(
            f"[adaptive membrane] Cell {cid}: "
            "skipped because too few valid voxels "
            "were available"
        )

        # Move to the next cell.
        continue

    # Calculate the total range of cell-boundary distances.
    #
    # np.ptp means:
    # maximum value minus minimum value.
    if np.ptp(d) <= 0:

        # All distances are identical, so a distance-based
        # intensity profile cannot be created.
        print(
            f"[adaptive membrane] Cell {cid}: "
            "skipped because the distance range is zero"
        )

        # Move to the next cell.
        continue

    # ------------------------------------------------------
    # AUTOMATIC BIN EDGES USING FREEDMAN–DIACONIS
    #
    # np.histogram_bin_edges supports bins="fd".
    #
    # scipy.stats.binned_statistic does not directly accept
    # the string "fd", so the edges are created first here.
    # ------------------------------------------------------
    try:

        # Automatically calculate distance-bin boundaries.
        edges = np.histogram_bin_edges(
            d,
            bins="fd"
        )

    except Exception as e:

        # Report failure to calculate Freedman–Diaconis bins.
        print(
            f"[adaptive membrane] Cell {cid}: "
            f"FD bin calculation failed: {e}"
        )

        # Skip this cell.
        continue

    # Convert bin edges into a float64 NumPy array.
    edges = np.asarray(
        edges,
        dtype=np.float64
    )

    # Keep only finite edge values.
    edges = edges[
        np.isfinite(edges)
    ]

    # Remove duplicate bin boundaries.
    #
    # Duplicate edges would create zero-width bins.
    edges = np.unique(
        edges
    )

    # Read the configured minimum number of radial bins.
    minimum_bins = MEMBRANE_MINIMUM_BINS

    # A profile with N bins requires N + 1 boundary values.
    if edges.size < minimum_bins + 1:

        # Replace the automatically calculated edges with
        # evenly spaced boundaries covering the full distance range.
        edges = np.linspace(
            float(d.min()),
            float(d.max()),
            minimum_bins + 1
        )

    # Read the configured maximum number of radial bins.
    maximum_bins = MEMBRANE_MAXIMUM_BINS

    # Check whether the number of bins exceeds the limit.
    #
    # Number of bins = number of edges minus one.
    if edges.size - 1 > maximum_bins:

        # Replace the edges with a limited number of evenly
        # spaced distance bins.
        edges = np.linspace(
            float(d.min()),
            float(d.max()),
            maximum_bins + 1
        )

    # A minimum of three edges is required to define
    # at least two bins.
    if edges.size < 3:

        # Explain why this cell cannot be processed.
        print(
            f"[adaptive membrane] Cell {cid}: "
            "skipped because valid bin edges "
            "could not be created"
        )

        # Move to the next cell.
        continue

    # ------------------------------------------------------
    # CALCULATE THE RADIAL MEAN-INTENSITY PROFILE
    # ------------------------------------------------------
    try:

        # Group voxels according to their boundary distance.
        #
        # For every radial bin, calculate the mean channel-2
        # intensity of voxels assigned to that bin.
        profile, edges, binnumber = binned_statistic(
            d,
            intensity,
            statistic="mean",
            bins=edges
        )

    except Exception as e:

        # Report profile-calculation failure.
        print(
            f"[adaptive membrane] Cell {cid}: "
            f"binned_statistic failed: {e}"
        )

        # Skip the current cell.
        continue

    # Convert the resulting mean-intensity profile
    # into a float64 array.
    profile = np.asarray(
        profile,
        dtype=np.float64
    )

    # Calculate the physical center of every distance bin.
    #
    # Bin centers are used for interpolation of peak widths.
    bin_centers = 0.5 * (
        edges[:-1]
        + edges[1:]
    )

    # ------------------------------------------------------
    # REMOVE EMPTY OR INVALID PROFILE BINS
    # ------------------------------------------------------

    # Identify bins where both the mean intensity
    # and bin-center distance are finite.
    good = (
        np.isfinite(profile)
        & np.isfinite(bin_centers)
    )

    # Require at least five valid profile bins.
    if good.sum() < 5:

        # Report the number of valid bins remaining.
        print(
            f"[adaptive membrane] Cell {cid}: "
            f"skipped because only "
            f"{int(good.sum())} valid bins remained"
        )

        # Move to the next cell.
        continue

    # Keep only valid mean-intensity values.
    profile = profile[
        good
    ]

    # Keep the corresponding valid distance-bin centers.
    bin_centers = bin_centers[
        good
    ]

    # ------------------------------------------------------
    # SMOOTH THE RADIAL INTENSITY PROFILE
    # ------------------------------------------------------

    # Apply one-dimensional Gaussian smoothing.
    #
    # sigma is expressed in profile-bin units.
    #
    # mode="nearest" extends the endpoint values when
    # calculating the filter near the profile boundaries.
    profile_smooth = gaussian_filter1d(
        profile,
        sigma=MEMBRANE_SIGMA_SMOOTH,
        mode="nearest"
    )

    # Verify that every smoothed intensity is finite.
    if not np.all(
        np.isfinite(profile_smooth)
    ):

        # Skip profiles containing NaN or infinite values.
        print(
            f"[adaptive membrane] Cell {cid}: "
            "skipped because the smoothed profile "
            "contains invalid values"
        )

        # Move to the next cell.
        continue

    # Check whether the smoothed profile has any intensity range.
    if np.ptp(profile_smooth) <= 0:

        # A flat profile does not contain a detectable peak.
        print(
            f"[adaptive membrane] Cell {cid}: "
            "skipped because the intensity profile is flat"
        )

        # Move to the next cell.
        continue

    # ------------------------------------------------------
    # FIND THE MEMBRANE-INTENSITY PEAK
    # ------------------------------------------------------
    # Detect local maxima in the smoothed profile.
    #
    # peaks contains the indices of detected maxima.
    #
    # peak_properties contains additional measurements
    # requested or calculated by find_peaks.
    peaks, peak_properties = find_peaks(
        profile_smooth
    )

    # Check whether no ordinary local peaks were detected.
    if len(peaks) == 0:

        # As a fallback, find the global maximum.
        candidate_peak = int(
            np.argmax(profile_smooth)
        )


        # Accept the global maximum only when it is not
        # located at either endpoint of the profile.
        if (
            0 < candidate_peak
            < len(profile_smooth) - 1
        ):

            # Store the global maximum as a one-element
            # integer peak array.
            peaks = np.array(
                [candidate_peak],
                dtype=int
            )

        else:

            # Endpoint maxima do not have valid peak widths
            # on both sides.
            print(
                f"[adaptive membrane] Cell {cid}: "
                "skipped because no valid membrane "
                "peak was found"
            )

            # Move to the next cell.
            continue

    # Select the detected peak with the highest
    # smoothed intensity.
    peak = int(
        peaks[
            np.argmax(
                profile_smooth[peaks]
            )
        ]
    )

    # ------------------------------------------------------
    # MEASURE THE PEAK WIDTH
    # ------------------------------------------------------
    try:

        # Measure the width of the selected peak.
        #
        # rel_height controls the vertical level at which
        # the width is measured relative to peak prominence.
        widths, width_heights, left_ips, right_ips = peak_widths(
            profile_smooth,
            [peak],
            rel_height=MEMBRANE_REL_HEIGHT
        )

    except Exception as e:

        # Report peak-width calculation failure.
        print(
            f"[adaptive membrane] Cell {cid}: "
            f"peak-width calculation failed: {e}"
        )

        # Skip this cell.
        continue

    # Validate the interpolated left and right crossing indices.
    if (
        len(left_ips) == 0
        or len(right_ips) == 0
        or not np.isfinite(left_ips[0])
        or not np.isfinite(right_ips[0])
    ):

        # The peak width could not be converted into
        # a valid interval.
        print(
            f"[adaptive membrane] Cell {cid}: "
            "skipped because the membrane width was invalid"
        )

        # Move to the next cell.
        continue

    # Create an array containing the integer profile indices.
    #
    # This is used to interpolate fractional peak-width indices
    # into physical distances.
    profile_indices = np.arange(
        len(bin_centers),
        dtype=np.float64
    )

    # Convert the fractional left crossing index
    # into a physical distance from the cell boundary.
    left_dist = float(
        np.interp(
            left_ips[0],
            profile_indices,
            bin_centers
        )
    )

    # Convert the fractional right crossing index
    # into a physical distance from the cell boundary.
    right_dist = float(
        np.interp(
            right_ips[0],
            profile_indices,
            bin_centers
        )
    )

    # Prevent the left membrane boundary from falling
    # below the minimum available cell distance.
    left_dist = max(
        float(d.min()),
        left_dist
    )

    # Prevent the right membrane boundary from exceeding
    # the maximum available cell distance.
    right_dist = min(
        float(d.max()),
        right_dist
    )

    # Validate the physical membrane interval.
    if (
        not np.isfinite(left_dist)
        or not np.isfinite(right_dist)
        or right_dist <= left_dist
    ):

        # Report the invalid interval.
        print(
            f"[adaptive membrane] Cell {cid}: "
            f"invalid membrane interval "
            f"({left_dist:.4f}, {right_dist:.4f}) µm"
        )

        # Move to the next cell.
        continue

    # ------------------------------------------------------
    # BUILD THE MEMBRANE MASK FOR THIS CELL
    # ------------------------------------------------------

    # Select voxels that:
    #
    #   1. belong to the current cell,
    #   2. have boundary distance >= left_dist,
    #   3. have boundary distance <= right_dist.
    membrane = (
        cell
        & (dist >= left_dist)
        & (dist <= right_dist)
    )

    # Count the number of voxels in the detected
    # membrane interval.
    membrane_voxels = int(
        membrane.sum()
    )

    # Check whether the membrane mask is empty.
    if membrane_voxels == 0:

        # Report that the calculated interval produced no voxels.
        print(
            f"[adaptive membrane] Cell {cid}: "
            "skipped because the resulting membrane was empty"
        )

        # Move to the next cell.
        continue

    # Add the current cell's membrane voxels to the
    # combined adaptive membrane mask.
    #
    # The |= operator performs an in-place Boolean OR.
    adaptive_membrane_mask |= membrane


    # Print diagnostic information for the current cell.
    print(
        f"[adaptive membrane] Cell {cid}: "

        # Number of valid bins in the smoothed profile.
        f"bins={len(profile_smooth)}, "

        # Physical distance of the selected peak
        # from the cell boundary.
        f"peak_distance={bin_centers[peak]:.4f} µm, "

        # Physical distance interval selected as membrane.
        f"interval={left_dist:.4f}-{right_dist:.4f} µm, "

        # Number of voxels in the final membrane mask.
        f"membrane_voxels={membrane_voxels}"
    )

# ============================================================
# FINAL MEMBRANE MASK
#
# Use the membrane detected from the adaptive radial-intensity
# analysis as the mask for signal and volume quantification.
# ============================================================
# Store the combined adaptive membrane mask under the
# general membrane-mask variable.
membrane_mask = adaptive_membrane_mask

# Use the membrane mask as the region for later quantification.
neuron_mask_for_quantification = membrane_mask


# Print the total number of foreground voxels in the
# adaptive membrane mask.
print(
    "[adaptive membrane] Total membrane voxels:",
    int(neuron_mask_for_quantification.sum())
)

# Check whether the adaptive membrane algorithm produced
# an entirely empty mask.
if not np.any(neuron_mask_for_quantification):

    # Warn that the membrane mask is empty.
    print(
        "[WARNING] The adaptive membrane mask is empty. "
        "Falling back to the filled cell segmentation mask."
    )

    # Use the complete filled-cell mask as a fallback.
    neuron_mask_for_quantification = (
        neuron_mask_for_segmentation.copy()
    )

    # Keep membrane_mask synchronized with the fallback mask.
    membrane_mask = neuron_mask_for_quantification

    # Print the number of foreground voxels in the fallback mask.
    print(
        "[adaptive membrane] Fallback mask voxels:",
        int(neuron_mask_for_quantification.sum())
    )

# ============================================================
# DEBUG EXPORTS
#
# Save intermediate masks and segmentation results so they
# can be inspected in ImageJ, Fiji, Napari, or another viewer.
# ============================================================
# Save the filled-cell Boolean mask as an 8-bit TIFF.
#
# False becomes 0.
# True becomes 255.
tiff.imwrite(
    outpath("debug_filled_cell_mask.tif"),
    neuron_mask_for_segmentation.astype(np.uint8) * 255
)

# Save the membrane mask used for quantification.
#
# If adaptive membrane detection failed, this file contains
# the fallback filled-cell mask instead.
tiff.imwrite(
    outpath("debug_adaptive_membrane_mask.tif"),
    neuron_mask_for_quantification.astype(np.uint8) * 255
)

# Save the internal distance-transform image.
#
# Values are stored as 32-bit floating point.
tiff.imwrite(
    outpath("debug_dist_inside.tif"),
    dist_inside.astype(np.float32)
)

# Save the watershed marker image.
#
# Marker labels are stored as unsigned 16-bit integers.
tiff.imwrite(
    outpath("debug_markers.tif"),
    markers.astype(np.uint16)
)

# Save the final watershed cell-label image.
tiff.imwrite(
    outpath("debug_cell_seg.tif"),
    cell_seg.astype(np.uint16)
)

# Confirm that all debugging images were saved.
print(
    "Saved debug filled-cell mask, adaptive membrane mask, "
    "distance map, watershed markers, and cell segmentation."
)

# ============================================================
# VISUALIZATION-ONLY CELL FILTERING
#
# Remove very small cells from visualizations and replace
# original watershed labels with consecutive serial IDs.
#
# This does not directly modify the original cell_seg array.
# ============================================================
# Copy the original cell segmentation for visualization.
cell_seg_viz = cell_seg.copy()

# Create a dictionary that maps:
#
# original watershed ID -> visualization serial ID.
cell_id_map_viz = {}

# Verify that cell_seg_viz is a NumPy array and contains
# at least one foreground label.
if (
    isinstance(cell_seg_viz, np.ndarray)
    and cell_seg_viz.max() > 0
):
    # Count the number of voxels belonging to every label.
    #
    # Array index = label ID.
    # Array value = voxel count for that label.
    counts_viz = np.bincount(
        cell_seg_viz.ravel().astype(np.int64)
    )

    # Find labels whose voxel count is smaller than the
    # configured visualization threshold.
    tiny_labels = np.where(
        counts_viz < VIZ_MIN_VOXELS
    )[0]

    # Remove label 0 because it represents background.
    tiny_labels = tiny_labels[
        tiny_labels > 0
    ]

    # Check whether any small labels were found.
    if tiny_labels.size > 0:

        # Replace every tiny cell label with background.
        cell_seg_viz[
            np.isin(cell_seg_viz, tiny_labels)
        ] = 0

    # Find all remaining unique labels.
    unique_labels = np.unique(
        cell_seg_viz
    )

    # Remove the background label.
    unique_labels = unique_labels[
        unique_labels > 0
    ]

    # Continue if at least one visible cell remains.
    if unique_labels.size > 0:

        # Allocate a new label image for consecutive IDs.
        new_seg = np.zeros_like(
            cell_seg_viz,
            dtype=np.int32
        )

        # Assign serial IDs beginning at 1.
        for new_id, old_id in enumerate(
            unique_labels,
            start=1
        ):

            # Replace the original label with the serial label.
            new_seg[
                cell_seg_viz == old_id
            ] = new_id

            # Save the original-to-serial ID relationship.
            cell_id_map_viz[
                int(old_id)
            ] = int(new_id)

        # Replace the visualization segmentation
        # with the consecutively relabeled image.
        cell_seg_viz = new_seg

# Create a Boolean visualization mask.
#
# True means a visible cell label is present.
cell_mask_viz = (
    cell_seg_viz > 0
)

# ============================================================
# DISTANCE- AND OVERLAP-AWARE CLASSIFICATION HELPERS
#
# These functions help classify each lysosome as:
#   - inside a cell
#   - outside all cells
# ============================================================
# Calculate distance from every outside voxel to the nearest
# foreground voxel of the original neuron mask.
#
# Because sampling is supplied, distances are in micrometers.
dist_out_um = edt(
    ~neuron_mask,
    sampling=(
        vz_um,
        vy_um,
        vx_um
    )
).astype(np.float32)

# Create a softened cell mask.
#
# It contains:
#   1. the original neuron mask, and
#   2. outside voxels within MARGIN_UM of that mask.
soft_cell_mask = (
    neuron_mask
    | (dist_out_um <= MARGIN_UM)
)

def nearest_cell_label(
    z,
    y,
    x,
    max_r=NEIGHBOR_MAX_VOX
):
    """
    Search progressively larger neighborhoods around one voxel
    and return the most common nonzero cell label.

    Returns zero when no nearby cell label is found.
    """

    # Read the dimensions of the watershed cell-label image.
    Z, Y, X = cell_seg.shape

    # Search radii from one voxel up to max_r voxels.
    for r in range(
        1,
        max_r + 1
    ):

        # Calculate the lower and upper Z bounds.
        z1 = max(
            0,
            z - r
        )
        z2 = min(
            Z,
            z + r + 1
        )

        # Calculate the lower and upper Y bounds.
        y1 = max(
            0,
            y - r
        )
        y2 = min(
            Y,
            y + r + 1
        )

        # Calculate the lower and upper X bounds.
        x1 = max(
            0,
            x - r
        )
        x2 = min(
            X,
            x + r + 1
        )

        # Extract the local cell-label region.
        patch = cell_seg[
            z1:z2,
            y1:y2,
            x1:x2
        ]

        # Keep only non-background labels.
        lab = patch[
            patch > 0
        ]

        # Check whether the neighborhood contains cell labels.
        if lab.size:

            # Count occurrences of each label and return
            # the most frequent label.
            return int(
                np.bincount(
                    lab.ravel()
                ).argmax()
            )

    # Return zero when no cell was found nearby.
    return 0

def sphere_overlap_fraction(
    zc_um,
    yc_um,
    xc_um,
    r_um,
    mask_bool
):
    """
    Calculate what fraction of a physical 3D sphere overlaps
    a supplied Boolean mask.
    """

    # A zero or negative radius cannot define a valid sphere.
    if r_um <= 0:
        return 0.0

    # Convert the physical Z center into a voxel coordinate.
    zc = int(
        round(zc_um / vz_um)
    )

    # Convert the physical Y center into a voxel coordinate.
    yc = int(
        round(yc_um / vy_um)
    )

    # Convert the physical X center into a voxel coordinate.
    xc = int(
        round(xc_um / vx_um)
    )

    # Convert the physical radius into a Z half-window.
    rz = max(
        1,
        int(np.ceil(r_um / vz_um))
    )

    # Convert the physical radius into a Y half-window.
    ry = max(
        1,
        int(np.ceil(r_um / vy_um))
    )

    # Convert the physical radius into an X half-window.
    rx = max(
        1,
        int(np.ceil(r_um / vx_um))
    )

    # Read the mask dimensions.
    Z, Y, X = mask_bool.shape

    # Calculate clipped Z extraction bounds.
    z1 = max(
        0,
        zc - rz
    )
    z2 = min(
        Z,
        zc + rz + 1
    )

    # Calculate clipped Y extraction bounds.
    y1 = max(
        0,
        yc - ry
    )
    y2 = min(
        Y,
        yc + ry + 1
    )

    # Calculate clipped X extraction bounds.
    x1 = max(
        0,
        xc - rx
    )
    x2 = min(
        X,
        xc + rx + 1
    )

    # Reject invalid or empty extraction regions.
    if (
        z1 >= z2
        or y1 >= y2
        or x1 >= x2
    ):
        return 0.0

    # Generate coordinate grids for the local region.
    zz, yy, xx = np.mgrid[
        z1:z2,
        y1:y2,
        x1:x2
    ]

    # Calculate Z distances from the sphere center in µm.
    dz = (
        zz - zc
    ) * vz_um

    # Calculate Y distances from the sphere center in µm.
    dy = (
        yy - yc
    ) * vy_um

    # Calculate X distances from the sphere center in µm.
    dx = (
        xx - xc
    ) * vx_um

    # Create a physical 3D spherical mask.
    sphere = (
        dz * dz
        + dy * dy
        + dx * dx
    ) <= (
        r_um * r_um
    )

    # Return zero if the discretized sphere contains no voxels.
    if not np.any(sphere):
        return 0.0

    # Select sphere voxels that also lie inside mask_bool.
    in_mask = (
        mask_bool[
            z1:z2,
            y1:y2,
            x1:x2
        ]
        & sphere
    )

    # Return the fraction of sphere voxels inside the mask.
    return (
        float(in_mask.sum())
        / float(sphere.sum())
    )

# ============================================================
# MAP LYSOSOMES TO CELL OR OUTSIDE
#
# Classification uses:
#   1. hard center inclusion,
#   2. soft-margin center inclusion,
#   3. sphere-overlap fraction.
# ============================================================

# List that will store "cell" or "outside".
location_ch2 = []

# List that will store assigned watershed cell IDs.
cell_id_list = []

# Restore the complete lysosome dataset.
df = df_all.copy()


# Continue only when lysosome rows exist.
if len(df) > 0:

    # Read the dimensions of the original neuron mask.
    Z, Y, X = neuron_mask.shape

    # Process each lysosome's position and radius.
    for (
        zc_um,
        yc_um,
        xc_um,
        r_um
    ) in df[
        [
            "z_um",
            "y_um",
            "x_um",
            "radius_um"
        ]
    ].to_numpy():

        # Convert physical Z position to a voxel index.
        zz = int(
            round(zc_um / vz_um)
        )

        # Convert physical Y position to a voxel index.
        yy = int(
            round(yc_um / vy_um)
        )

        # Convert physical X position to a voxel index.
        xx = int(
            round(xc_um / vx_um)
        )

        # Test whether the lysosome center lies directly
        # inside the original cell mask.
        inside_hard = (
            0 <= zz < Z
            and 0 <= yy < Y
            and 0 <= xx < X
            and neuron_mask[zz, yy, xx]
        )

        # Test whether the center lies in the expanded soft mask.
        inside_soft = (
            0 <= zz < Z
            and 0 <= yy < Y
            and 0 <= xx < X
            and soft_cell_mask[zz, yy, xx]
        )

        # Begin with the hard-mask classification.
        is_inside = bool(
            inside_hard
        )

        # Accept soft-mask inclusion when hard inclusion failed.
        if not is_inside and inside_soft:
            is_inside = True

        # If the center is still outside, test sphere overlap.
        if not is_inside:

            # Calculate the fraction of the lysosome sphere
            # overlapping the original neuron mask.
            frac = sphere_overlap_fraction(
                zc_um,
                yc_um,
                xc_um,
                r_um,
                neuron_mask
            )

            # Classify the lysosome as inside when the overlap
            # reaches the configured threshold.
            if frac >= OVERLAP_ALPHA:
                is_inside = True

        # Handle lysosomes classified as inside a cell.
        if is_inside:

            # Initialize the assigned cell ID as zero.
            cid = 0

            # Check that the lysosome center lies inside the image.
            if (
                0 <= zz < Z
                and 0 <= yy < Y
                and 0 <= xx < X
            ):

                # Use the direct watershed label when available.
                #
                # Otherwise search for the nearest local cell label.
                cid = (
                    int(cell_seg[zz, yy, xx])
                    if cell_seg[zz, yy, xx] != 0
                    else nearest_cell_label(
                        zz,
                        yy,
                        xx
                    )
                )

            # Record the intracellular classification.
            location_ch2.append(
                "cell"
            )

            # Record the assigned cell ID.
            cell_id_list.append(
                cid
            )

        else:

            # Record an extracellular lysosome.
            location_ch2.append(
                "outside"
            )

            # Outside lysosomes receive cell ID zero.
            cell_id_list.append(
                0
            )


    # Add the location classification to the DataFrame.
    df["location_ch2"] = location_ch2

    # Add the original watershed cell assignment.
    df["cell_id_ch2"] = cell_id_list

    # Map original watershed IDs to serial visualization IDs.
    #
    # Cells removed from visualization receive ID zero.
    df["cell_id_ch2_viz"] = (
        df["cell_id_ch2"]
        .map(cell_id_map_viz)
        .fillna(0)
        .astype(int)
    )

    # Count lysosomes classified as cell versus outside
    # and save the summary table.
    df.groupby(
        "location_ch2"
    ).size().reset_index(
        name="count"
    ).to_csv(
        outpath(
            "lysosome_counts_cell_vs_outside.csv"
        ),
        index=False
    )

    # Count intracellular lysosomes for every original cell ID.
    (
        df[
            df["location_ch2"] == "cell"
        ]
        .groupby(
            "cell_id_ch2"
        )
        .size()
        .reset_index(
            name="count"
        )
        .to_csv(
            outpath(
                "lysosome_counts_by_cell.csv"
            ),
            index=False
        )
    )

    # Create a per-cell serial lysosome-ID column.
    #
    # Zero means the lysosome is outside or unassigned.
    df["lys_id_in_cell"] = 0


    # Select intracellular lysosomes with valid cell IDs.
    mask_in = (
        (df["location_ch2"] == "cell")
        & (df["cell_id_ch2"] > 0)
    )

    # Sort intracellular lysosomes consistently within each cell.
    df_sorted = (
        df.loc[
            mask_in
        ]
        .sort_values(
            [
                "cell_id_ch2",
                "z_um",
                "y_um",
                "x_um"
            ]
        )
        .copy()
    )

    # Assign serial lysosome IDs beginning at 1 within each cell.
    df.loc[
        df_sorted.index,
        "lys_id_in_cell"
    ] = (
        df_sorted
        .groupby(
            "cell_id_ch2"
        )
        .cumcount()
        .to_numpy()
        + 1
    ).astype(int)

    # Save the complete lysosome-to-cell assignment table.
    df.to_csv(
        outpath(
            "lysosomes_with_cell_vs_outside.csv"
        ),
        index=False
    )

    # Determine the maximum per-cell lysosome serial ID.
    #
    # This is equivalent to the number of assigned lysosomes
    # when IDs are consecutive.
    lys_serial_counts = (
        df[
            df["lys_id_in_cell"] > 0
        ]
        .groupby(
            "cell_id_ch2"
        )[
            "lys_id_in_cell"
        ]
        .max()
        .reset_index()
        .rename(
            columns={
                "lys_id_in_cell":
                    "lysosomes_in_cell"
            }
        )
    )

    # --------------------------------------------------------
    # COMPUTE CELL VOLUME FOR EACH SERIAL VISUALIZATION ID
    # --------------------------------------------------------
    # Count the voxels belonging to every visualization label.
    counts_serial = np.bincount(
        cell_seg_viz.ravel().astype(np.int64)
    )

    # Convert label voxel counts into physical volumes.
    vol_serial_um3 = (
        counts_serial.astype(np.float64)
        * float(voxel_um3)
    )

    # Map original watershed IDs to serial visualization IDs.
    lys_serial_counts["cell_id_serial"] = (
        lys_serial_counts[
            "cell_id_ch2"
        ]
        .map(cell_id_map_viz)
        .fillna(0)
        .astype(int)
    )

    # Assign the physical cell volume associated with each
    # serial visualization ID.
    lys_serial_counts["cell_volume_um3"] = (
        lys_serial_counts[
            "cell_id_serial"
        ]
        .apply(
            lambda sid: (
                float(vol_serial_um3[sid])
                if (
                    sid > 0
                    and sid < len(vol_serial_um3)
                )
                else 0.0
            )
        )
    )

    # Save per-cell lysosome counts and cell volumes.
    lys_serial_counts.to_csv(
        outpath(
            "lysosome_counts_by_cell_serial.csv"
        ),
        index=False
    )

# Print all expected lysosome classification outputs.
#
# Note: when df is empty, some of these files may not have
# been created even though their paths are printed.
print(
    "Saved:",
    outpath(
        "lysosome_counts_cell_vs_outside.csv"
    ),
    outpath(
        "lysosome_counts_by_cell.csv"
    ),
    outpath(
        "lysosomes_with_cell_vs_outside.csv"
    ),
    outpath(
        "lysosome_counts_by_cell_serial.csv"
    )
)

# ============================================================
# SIGNAL AND VOLUME QUANTIFICATION
# ============================================================
# Announce the beginning of signal quantification.
print(
    "\n[Signal quantification] Starting..."
)

# Convert raw channel 1 to 32-bit floating point.
signal_img = img_ch1.astype(
    np.float32
)

# Print the complete intensity array.
#
# Warning: this can produce extremely large notebook output.
print(
    f"Voxel intensities: {signal_img}"
)

# ------------------------------------------------------------
# BACKGROUND SUBTRACTION
# ------------------------------------------------------------

# Estimate background as the fifth intensity percentile.
bg = np.percentile(
    signal_img,
    5
)

# Subtract the estimated background and prevent
# negative corrected intensities.
signal_img = np.clip(
    signal_img - bg,
    0,
    None
)

# Print the complete background-subtracted array.
#
# This can also produce very large output.
print(
    f"Subtracts background from every voxel: {signal_img}"
)

# ------------------------------------------------------------
# GET VALID CELL IDS
# ------------------------------------------------------------

# Print the complete watershed segmentation array.
print(
    f"The watershed segmentation image: {cell_seg}"
)

# Extract every unique cell label, including background zero.
cell_ids = np.unique(
    cell_seg
)

# Print all unique labels.
print(
    f"Unique labels: {cell_ids}"
)

# Remove background label zero.
cell_ids = cell_ids[
    cell_ids > 0
]

# Print the remaining cell labels.
print(
    f"Unique labels removing background: {cell_ids}"
)

# Print the number of detected cells.
print(
    f"Number of cells: {len(cell_ids)}"
)

# ------------------------------------------------------------
# TOTAL SIGNAL AND VOLUME PER CELL
# ------------------------------------------------------------
# Dictionary mapping cell ID to total channel-1 signal.
cell_signal = {}

# Dictionary mapping cell ID to quantified voxel count.
cell_volume_vox = {}

# Process each valid cell ID.
for cid in cell_ids:
    # Select voxels that belong to this watershed cell
    # and are also inside the quantification mask.
    cell_mask = (
        (cell_seg == cid)
        & neuron_mask_for_quantification
    )

    # Sum background-corrected channel-1 signal in the mask.
    cell_signal[cid] = float(
        signal_img[
            cell_mask
        ].sum()
    )

    # Count voxels in the quantified cell region.
    cell_volume_vox[cid] = int(
        cell_mask.sum()
    )

    # Print the quantified voxel count and signal.
    print(
        f"[CELL {cid}] "
        f"voxels={cell_volume_vox[cid]} "
        f"signal={cell_signal[cid]}"
    )

# ============================================================
# HELPER: PHYSICAL 3D SPHERE MASK
# ============================================================
def sphere_mask(
    zc_um,
    yc_um,
    xc_um,
    r_um
):
    """
    Create a local Boolean sphere centered at a position
    supplied in micrometers.
    """
    # Convert physical Z coordinate to a voxel index.
    zc = int(
        round(zc_um / vz_um)
    )

    # Convert physical Y coordinate to a voxel index.
    yc = int(
        round(yc_um / vy_um)
    )

    # Convert physical X coordinate to a voxel index.
    xc = int(
        round(xc_um / vx_um)
    )

    # Convert physical radius to a Z half-window.
    rz = max(
        1,
        int(np.ceil(r_um / vz_um))
    )

    # Convert physical radius to a Y half-window.
    ry = max(
        1,
        int(np.ceil(r_um / vy_um))
    )

    # Convert physical radius to an X half-window.
    rx = max(
        1,
        int(np.ceil(r_um / vx_um))
    )

    # Read the signal-image dimensions.
    Z, Y, X = signal_img.shape

    # Calculate clipped local Z bounds.
    z1 = max(
        0,
        zc - rz
    )
    z2 = min(
        Z,
        zc + rz + 1
    )

    # Calculate clipped local Y bounds.
    y1 = max(
        0,
        yc - ry
    )
    y2 = min(
        Y,
        yc + ry + 1
    )

    # Calculate clipped local X bounds.
    x1 = max(
        0,
        xc - rx
    )
    x2 = min(
        X,
        xc + rx + 1
    )

    # Generate coordinate grids for the local bounding box.
    zz, yy, xx = np.mgrid[
        z1:z2,
        y1:y2,
        x1:x2
    ]

    # Calculate physical Z displacement from the center.
    dz = (
        zz - zc
    ) * vz_um

    # Calculate physical Y displacement from the center.
    dy = (
        yy - yc
    ) * vy_um

    # Calculate physical X displacement from the center.
    dx = (
        xx - xc
    ) * vx_um

    # Create the physical spherical mask.
    sphere = (
        dz**2
        + dy**2
        + dx**2
    ) <= (
        r_um**2
    )

    # Return bounds and the local sphere mask.
    return (
        z1,
        z2,
        y1,
        y2,
        x1,
        x2,
        sphere
    )

# ============================================================
# LYSOSOME CORE MASK
# ============================================================
# Allocate a global Boolean mask for all intracellular
# lysosome-core spheres.
lys_mask = np.zeros_like(
    signal_img,
    dtype=bool
)

# Process every lysosome row.
for _, row in df.iterrows():

    # Ignore lysosomes classified outside cells.
    if row["location_ch2"] != "cell":
        continue

    # Build the local sphere for this lysosome.
    (
        z1,
        z2,
        y1,
        y2,
        x1,
        x2,
        sphere
    ) = sphere_mask(
        row["z_um"],
        row["y_um"],
        row["x_um"],
        row["radius_um"]
    )

    # Add the sphere to the combined lysosome-core mask.
    lys_mask[
        z1:z2,
        y1:y2,
        x1:x2
    ] |= sphere

# ============================================================
# MASK 2: LYSOSOME-ASSOCIATED REGION
#
# This mask contains the lysosome core plus a surrounding
# physical halo.
# ============================================================
# Define the halo thickness in micrometers.
HALO_UM = 0.4

# Calculate physical distance from every non-core voxel
# to the nearest lysosome-core voxel.
dist_to_lys_um = distance_transform_edt(
    ~lys_mask,
    sampling=(
        vz_um,
        vy_um,
        vx_um
    )
).astype(np.float32)

# Select all voxels located within HALO_UM of a core.
lys_assoc_mask = (
    dist_to_lys_um <= HALO_UM
)

# ============================================================
# MASK 1: MEMBRANE
# ============================================================
# Assign the final membrane mask.
mask1 = membrane_mask

# ============================================================
# MASK 2: LYSOSOME CORE PLUS ASSOCIATED REGION
# ============================================================

# Assign the lysosome-associated mask.
mask2 = lys_assoc_mask

# ============================================================
# MASK 3: MEMBRANE MINUS LYSOSOME-ASSOCIATED REGION
# ============================================================
# Keep membrane voxels that do not belong to mask2.
mask3 = (
    mask1
    & (~mask2)
)

# Preserve the previous residual-mask variable name.
residual_mask = mask3

# ============================================================
# 2D PER-SLICE OUTLINES
#
# Generate outlines independently in every XY slice.
# ============================================================
# Set the outline thickness in XY pixels.
OUTLINE_THICKNESS_PX = 1

# Create the two-dimensional disk footprint.
outline_footprint = disk(
    OUTLINE_THICKNESS_PX
)

# Allocate a 3D Boolean stack for lysosome-core outlines.
lys_core_outline_2d = np.zeros_like(
    lys_mask,
    dtype=bool
)

# Allocate a 3D Boolean stack for associated-region outlines.
lys_assoc_outline_2d = np.zeros_like(
    lys_assoc_mask,
    dtype=bool
)

# Process every Z slice independently.
for z in range(
    lys_mask.shape[0]
):
    # Extract the current lysosome-core slice.
    core_z = lys_mask[
        z
    ].astype(bool)

    # Extract the current associated-region slice.
    assoc_z = lys_assoc_mask[
        z
    ].astype(bool)


    # Create the core outline only when core pixels exist.
    if np.any(core_z):

        # Erode the core using the two-dimensional footprint.
        core_eroded_z = binary_erosion(
            core_z,
            footprint=outline_footprint
        )

        # Keep original pixels removed by erosion.
        lys_core_outline_2d[z] = (
            core_z
            & ~core_eroded_z
        )

    # Create the associated-region outline when pixels exist.
    if np.any(assoc_z):

        # Erode the associated region.
        assoc_eroded_z = binary_erosion(
            assoc_z,
            footprint=outline_footprint
        )

        # Keep only boundary pixels removed by erosion.
        lys_assoc_outline_2d[z] = (
            assoc_z
            & ~assoc_eroded_z
        )

# Print outline voxel counts.
print(
    "[2D lysosome outlines] "
    f"core outline voxels="
    f"{int(lys_core_outline_2d.sum())}, "
    f"associated outline voxels="
    f"{int(lys_assoc_outline_2d.sum())}"
)

# ============================================================
# SAVE BINARY OUTLINE STACKS
# ============================================================
# Save the core outline as an 8-bit ZYX TIFF.
tiff.imwrite(
    outpath(
        "OUTLINE_2D_lysosome_core.tif"
    ),
    lys_core_outline_2d.astype(np.uint8) * 255,
    metadata={
        "axes": "ZYX"
    }
)

# Save the associated-region outline as an 8-bit ZYX TIFF.
tiff.imwrite(
    outpath(
        "OUTLINE_2D_lysosome_associated.tif"
    ),
    lys_assoc_outline_2d.astype(np.uint8) * 255,
    metadata={
        "axes": "ZYX"
    }
)

# Print the saved core-outline path.
print(
    "Saved:",
    outpath(
        "OUTLINE_2D_lysosome_core.tif"
    )
)

# Print the saved associated-outline path.
print(
    "Saved:",
    outpath(
        "OUTLINE_2D_lysosome_associated.tif"
    )
)

# ============================================================
# RGB RAW-DATA OVERLAY
#
# Raw channel 1       = grayscale
# Lysosome core       = yellow outline
# Associated region   = magenta outline
# ============================================================
# Normalize raw channel 1 to uint8.
raw_ch1_u8 = _norm_u8_stack(
    img_ch1.astype(np.float32)
)

# Read normalized image dimensions.
Z, H, W = raw_ch1_u8.shape


# Allocate the final RGB stack.
lysosome_outline_rgb = np.zeros(
    (
        Z,
        H,
        W,
        3
    ),
    dtype=np.uint8
)

# Process every Z slice.
for z in range(Z):
    # Extract the grayscale slice.
    gray_z = raw_ch1_u8[
        z
    ]

    # Replicate grayscale into red, green, and blue channels.
    rgb_z = np.dstack(
        [
            gray_z,
            gray_z,
            gray_z
        ]
    ).astype(np.float32)

    # Extract the core outline for this slice.
    core_outline_z = lys_core_outline_2d[
        z
    ]

    # Extract the associated-region outline.
    assoc_outline_z = lys_assoc_outline_2d[
        z
    ]

    # Color associated-region outline pixels magenta.
    rgb_z[
        assoc_outline_z,
        0
    ] = 255

    rgb_z[
        assoc_outline_z,
        1
    ] = 0

    rgb_z[
        assoc_outline_z,
        2
    ] = 255

    # Color core-outline pixels yellow.
    #
    # These assignments occur after magenta, so yellow
    # remains visible where both outlines overlap.
    rgb_z[
        core_outline_z,
        0
    ] = 255

    rgb_z[
        core_outline_z,
        1
    ] = 255

    rgb_z[
        core_outline_z,
        2
    ] = 0

    # Clip values and save the finished slice.
    lysosome_outline_rgb[z] = np.clip(
        rgb_z,
        0,
        255
    ).astype(np.uint8)

# Save the RGB outline TIFF stack.
tiff.imwrite(
    outpath(
        "RAW_Ch1_with_2D_lysosome_outlines.tif"
    ),
    lysosome_outline_rgb,
    photometric="rgb",
    metadata={
        "axes": "ZYXS"
    }
)

# Print the saved overlay path.
print(
    "Saved:",
    outpath(
        "RAW_Ch1_with_2D_lysosome_outlines.tif"
    )
)

# ============================================================
# GREEN CHANNEL: MEMBRANE PLUS CORE LYSOSOMES
# ============================================================
# Combine membrane and core masks.
green_membrane_plus_core_mask = (
    membrane_mask
    | lys_mask
)

# Retain raw channel-1 intensities only inside the
# membrane-plus-core mask.
green_membrane_plus_core_signal = np.where(
    green_membrane_plus_core_mask,
    img_ch1.astype(np.float32),
    0
).astype(np.float32)

# ============================================================
# SAVE DEBUG MASKS
# ============================================================

# Save membrane mask.
tiff.imwrite(
    outpath(
        "debug_mask1_membrane.tif"
    ),
    mask1.astype(np.uint8) * 255
)

# Save lysosome-associated mask.
tiff.imwrite(
    outpath(
        "debug_mask2_lysosome_associated.tif"
    ),
    mask2.astype(np.uint8) * 255
)

# Save residual membrane mask.
tiff.imwrite(
    outpath(
        "debug_mask3_membrane_minus_lysosomes.tif"
    ),
    mask3.astype(np.uint8) * 255
)

# Save physical distance-to-lysosome map.
tiff.imwrite(
    outpath(
        "debug_distance_to_lysosomes_um.tif"
    ),
    dist_to_lys_um.astype(np.float32)
)

# ============================================================
# NEW VISUALIZATION CHANNELS
#
# mask1 = membrane
# mask2 = lysosome core plus halo
# mask3 = membrane excluding mask2
# ============================================================

# Announce visualization-channel creation.
print(
    "\n[Membrane visualization] "
    "Creating new channels..."
)

# Convert channel 1 to float.
ch1_float = img_ch1.astype(
    np.float32
)

# Keep channel-1 intensity only where membrane exists.
ch1_membrane_overlap = np.where(
    mask1,
    ch1_float,
    0
).astype(np.float32)

# Keep channel-1 intensity only in residual membrane.
ch1_membrane_residual = np.where(
    mask3,
    ch1_float,
    0
).astype(np.float32)

# Convert membrane mask to an 8-bit binary image.
membrane_overlap_binary = mask1.astype(
    np.uint8
)

# Convert residual mask to an 8-bit binary image.
membrane_residual_binary = mask3.astype(
    np.uint8
)

# Save channel-1 signal inside the membrane mask.
tiff.imwrite(
    outpath(
        "CHANNEL_Ch1_inside_membrane_mask.tif"
    ),
    ch1_membrane_overlap,
    metadata={
        "axes": "ZYX",
        "PhysicalSizeX": float(vx_um),
        "PhysicalSizeY": float(vy_um),
        "PhysicalSizeZ": float(vz_um),
        "PhysicalSizeXUnit": "um",
        "PhysicalSizeYUnit": "um",
        "PhysicalSizeZUnit": "um",
    }
)

# Save residual membrane channel-1 signal.
tiff.imwrite(
    outpath(
        "CHANNEL_Ch1_membrane_minus_lysosomes.tif"
    ),
    ch1_membrane_residual,
    metadata={
        "axes": "ZYX",
        "PhysicalSizeX": float(vx_um),
        "PhysicalSizeY": float(vy_um),
        "PhysicalSizeZ": float(vz_um),
        "PhysicalSizeXUnit": "um",
        "PhysicalSizeYUnit": "um",
        "PhysicalSizeZUnit": "um",
    }
)

# Stack six useful channels along a new channel axis.
visualization_channels = np.stack(
    [
        # C0: raw channel 1.
        ch1_float,

        # C1: channel 1 inside membrane.
        ch1_membrane_overlap,

        # C2: residual channel 1.
        ch1_membrane_residual,

        # C3: binary membrane.
        mask1.astype(np.float32),

        # C4: binary lysosome-associated region.
        mask2.astype(np.float32),

        # C5: binary residual membrane.
        mask3.astype(np.float32),
    ],
    axis=0
)

# Save all channels as one OME-TIFF.
tiff.imwrite(
    outpath(
        "MULTICHANNEL_membrane_lysosome_results.ome.tif"
    ),
    visualization_channels,
    ome=True,
    metadata={
        "axes": "CZYX",

        # Assign descriptive channel names.
        "Channel": {
            "Name": [
                "Ch1 raw",
                "Ch1 inside membrane",
                "Ch1 membrane minus lysosomes",
                "Membrane binary mask",
                "Lysosome core plus diffuse binary mask",
                "Membrane minus lysosomes binary mask",
            ]
        },

        # Store physical voxel dimensions.
        "PhysicalSizeX": float(vx_um),
        "PhysicalSizeY": float(vy_um),
        "PhysicalSizeZ": float(vz_um),

        # Store physical-size units.
        "PhysicalSizeXUnit": "um",
        "PhysicalSizeYUnit": "um",
        "PhysicalSizeZUnit": "um",
    }
)

# Print the multichannel output path.
print(
    "Saved:",
    outpath(
        "MULTICHANNEL_membrane_lysosome_results.ome.tif"
    )
)

# ============================================================
# SLICE-BY-SLICE RGB VISUALIZATION
#
# Grayscale = raw channel 1
# Green     = membrane
# Red       = lysosome-associated region
# Magenta   = residual membrane
# ============================================================
# Normalize channel 1 for RGB visualization.
ch1_u8_visualization = _norm_u8_stack(
    ch1_float
)

# Read normalized stack dimensions.
Z, H, W = ch1_u8_visualization.shape


# Allocate the RGB visualization stack.
membrane_rgb_slices = np.zeros(
    (
        Z,
        H,
        W,
        3
    ),
    dtype=np.uint8
)

# Process every Z slice.
for z in range(Z):
    # Extract the normalized grayscale slice.
    gray = ch1_u8_visualization[
        z
    ]

    # Replicate grayscale into RGB.
    rgb = np.dstack(
        [
            gray,
            gray,
            gray
        ]
    ).astype(np.float32)

    # Extract membrane mask for this slice.
    membrane_z = mask1[
        z
    ]

    # Extract lysosome-associated mask.
    lysosome_z = mask2[
        z
    ]

    # Extract residual membrane mask.
    residual_z = mask3[
        z
    ]


    # Create a pure-green image.
    green_overlay = np.zeros_like(
        rgb
    )

    green_overlay[
        ...,
        1
    ] = 255


    # Blend green into membrane pixels.
    rgb[
        membrane_z
    ] = (
        0.55 * rgb[membrane_z]
        + 0.45 * green_overlay[membrane_z]
    )


    # Create a pure-red image.
    red_overlay = np.zeros_like(
        rgb
    )

    red_overlay[
        ...,
        0
    ] = 255


    # Blend red into lysosome-associated pixels.
    rgb[
        lysosome_z
    ] = (
        0.35 * rgb[lysosome_z]
        + 0.65 * red_overlay[lysosome_z]
    )


    # Create a magenta image.
    magenta_overlay = np.zeros_like(
        rgb
    )

    magenta_overlay[
        ...,
        0
    ] = 255

    magenta_overlay[
        ...,
        2
    ] = 255


    # Blend magenta strongly into residual-membrane pixels.
    rgb[
        residual_z
    ] = (
        0.20 * rgb[residual_z]
        + 0.80 * magenta_overlay[residual_z]
    )


    # Clip and save the final RGB slice.
    membrane_rgb_slices[z] = np.clip(
        rgb,
        0,
        255
    ).astype(np.uint8)

# Save the RGB slice stack.
tiff.imwrite(
    outpath(
        "SLICE_BY_SLICE_Ch1_membrane_and_residual_overlay.tif"
    ),
    membrane_rgb_slices,
    photometric="rgb",
    metadata={
        "axes": "ZYXS"
    }
)

# Print the saved RGB path.
print(
    "Saved:",
    outpath(
        "SLICE_BY_SLICE_Ch1_membrane_and_residual_overlay.tif"
    )
)

# ============================================================
# DISTANCE-TO-PUNCTA BINS
#
# Each tuple represents:
#   lower distance inclusive,
#   upper distance exclusive.
# ============================================================
DISTANCE_BINS_UM = [
    (0.0, 0.2),
    (0.2, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, np.inf),
]

# ============================================================
# CORTICAL OR PERIPHERAL ZONE
# ============================================================
# Define the depth of the peripheral zone in micrometers.
CORTEX_UM = 1.0

# ============================================================
# COMPUTE SIGNAL AND VOLUME PER CELL
# ============================================================
# Create a list that will contain one result dictionary
# for every cell.
rows = []

# Process every valid cell.
for cid in cell_ids:

    # Select the complete watershed region for this cell.
    cell_mask = (
        cell_seg == cid
    )

    # Select the portion of the cell included in the
    # membrane or fallback quantification mask.
    cell_quant_mask = (
        (cell_seg == cid)
        & neuron_mask_for_quantification
    )

    # Calculate total corrected signal in the quantified region.
    total_signal = float(
        signal_img[
            cell_quant_mask
        ].sum()
    )

    # Calculate quantified physical volume.
    total_vol = float(
        cell_quant_mask.sum()
        * voxel_um3
    )

    # --------------------------------------------------------
    # CORE LYSOSOME SIGNAL
    # --------------------------------------------------------

    # Select core lysosome voxels belonging to this cell.
    core_mask = (
        cell_mask
        & lys_mask
    )

    # Sum channel-1 signal inside core lysosomes.
    lysosome_core_signal = float(
        signal_img[
            core_mask
        ].sum()
    )

    # Calculate physical core volume.
    lysosome_core_volume_um3 = float(
        core_mask.sum()
        * voxel_um3
    )

    # --------------------------------------------------------
    # LYSOSOME-ASSOCIATED SIGNAL
    # --------------------------------------------------------
    # Select core-plus-halo voxels belonging to the cell.
    assoc_mask = (
        cell_mask
        & lys_assoc_mask
    )

    # Sum signal inside the associated region.
    lysosome_assoc_signal = float(
        signal_img[
            assoc_mask
        ].sum()
    )

    # Calculate associated-region physical volume.
    lysosome_assoc_volume_um3 = float(
        assoc_mask.sum()
        * voxel_um3
    )

    # --------------------------------------------------------
    # RESIDUAL NON-PUNCTA MEMBRANE SIGNAL
    # --------------------------------------------------------
    # Select residual membrane voxels belonging to the cell.
    membrane_residual_mask = (
        cell_mask
        & residual_mask
    )

    # Sum residual membrane signal.
    membrane_residual_signal = float(
        signal_img[
            membrane_residual_mask
        ].sum()
    )

    # Calculate residual membrane volume.
    membrane_residual_volume_um3 = float(
        membrane_residual_mask.sum()
        * voxel_um3
    )


    # Calculate the residual signal fraction.
    membrane_residual_fraction = (
        membrane_residual_signal
        / total_signal
        if total_signal > 0
        else np.nan
    )


    # Calculate the core-lysosome signal fraction.
    puncta_core_fraction = (
        lysosome_core_signal
        / total_signal
        if total_signal > 0
        else np.nan
    )


    # Calculate the associated-region signal fraction.
    puncta_associated_fraction = (
        lysosome_assoc_signal
        / total_signal
        if total_signal > 0
        else np.nan
    )

    # --------------------------------------------------------
    # CORTICAL OR PERIPHERAL ZONE
    # --------------------------------------------------------
    # Calculate physical distance from every cell voxel
    # to the nearest cell boundary.
    cell_dist_in_um = distance_transform_edt(
        cell_mask,
        sampling=(
            vz_um,
            vy_um,
            vx_um
        )
    ).astype(np.float32)

    # Select cell voxels within CORTEX_UM of the boundary.
    cortex_mask = (
        cell_mask
        & (
            cell_dist_in_um <= CORTEX_UM
        )
    )

    # Select cell voxels deeper than CORTEX_UM.
    inner_mask = (
        cell_mask
        & (
            cell_dist_in_um > CORTEX_UM
        )
    )

    # Select residual membrane in the cortical zone.
    cortex_residual_mask = (
        cortex_mask
        & residual_mask
    )

    # Select residual membrane in the inner zone.
    inner_residual_mask = (
        inner_mask
        & residual_mask
    )

    # Calculate mean residual signal in the cortex.
    residual_cortex_mean_HA = (
        float(
            signal_img[
                cortex_residual_mask
            ].mean()
        )
        if cortex_residual_mask.any()
        else np.nan
    )

    # Calculate mean residual signal in the inner region.
    residual_inner_mean_HA = (
        float(
            signal_img[
                inner_residual_mask
            ].mean()
        )
        if inner_residual_mask.any()
        else np.nan
    )

    # Calculate cortical enrichment relative to the inner region.
    residual_cortical_enrichment = (
        residual_cortex_mean_HA
        / residual_inner_mean_HA
        if (
            np.isfinite(
                residual_inner_mean_HA
            )
            and residual_inner_mean_HA > 0
        )
        else np.nan
    )

    # Create the result dictionary for this cell.
    row = {
        # Original watershed cell ID.
        "cell_id": int(cid),

        # Total signal and quantified cell volume.
        "cell_signal_total": total_signal,
        "cell_volume_um3": total_vol,

        # Core lysosome measurements.
        "lysosome_core_signal":
            lysosome_core_signal,

        "lysosome_core_volume_um3":
            lysosome_core_volume_um3,

        "puncta_core_fraction":
            puncta_core_fraction,

        # Core-plus-associated measurements.
        "lysosome_associated_signal":
            lysosome_assoc_signal,

        "lysosome_associated_volume_um3":
            lysosome_assoc_volume_um3,

        "puncta_associated_fraction":
            puncta_associated_fraction,

        # Residual membrane signal.
        "membrane_residual_magenta_signal":
            membrane_residual_signal,

        # Residual membrane physical volume.
        "membrane_residual_volume_um3":
            membrane_residual_volume_um3,

        # Signal per physical volume.
        #
        # Note: this is not the same as mean intensity per voxel.
        "membrane_residual_mean_magenta": (
            membrane_residual_signal
            / membrane_residual_volume_um3
            if membrane_residual_volume_um3 > 0
            else np.nan
        ),

        # Fraction of total signal in residual membrane.
        "membrane_residual_fraction":
            membrane_residual_fraction,
    }

    # --------------------------------------------------------
    # DISTANCE-TO-PUNCTA BINS
    # --------------------------------------------------------
    # Process each requested physical distance interval.
    for lo, hi in DISTANCE_BINS_UM:

        # Handle the final open-ended interval.
        if np.isinf(hi):

            # Select cell voxels at least lo micrometers
            # away from a lysosome core.
            bin_mask = (
                cell_mask
                & (
                    dist_to_lys_um >= lo
                )
            )

            # Build a text label for this interval.
            label_txt = (
                f"{lo:g}_plus"
            )

        else:

            # Select cell voxels in the half-open interval
            # lo <= distance < hi.
            bin_mask = (
                cell_mask
                & (
                    dist_to_lys_um >= lo
                )
                & (
                    dist_to_lys_um < hi
                )
            )

            # Build the interval label.
            label_txt = (
                f"{lo:g}_{hi:g}"
            )

        # Replace decimal points so the string is suitable
        # for use in a column name.
        label_txt = label_txt.replace(
            ".",
            "p"
        )

        # Calculate total signal in this distance bin.
        bin_signal = float(
            signal_img[
                bin_mask
            ].sum()
        )

        # Calculate physical volume of this distance bin.
        bin_volume_um3 = float(
            bin_mask.sum()
            * voxel_um3
        )

        # Save total signal under a dynamically generated name.
        row[
            f"HA_signal_{label_txt}_um_from_puncta"
        ] = bin_signal

        # Save physical volume.
        row[
            f"volume_{label_txt}_um_from_puncta"
        ] = bin_volume_um3

        # Save mean signal per voxel.
        row[
            f"mean_HA_{label_txt}_um_from_puncta"
        ] = (
            bin_signal
            / bin_mask.sum()
            if bin_mask.sum() > 0
            else np.nan
        )

    # Add this cell's result dictionary to the output list.
    rows.append(
        row
    )

    # Print a summary for the current cell.
    print(
        f"[CELL {cid}] "
        f"total_signal={total_signal:.2f} "
        f"core={lysosome_core_signal:.2f} "
        f"assoc={lysosome_assoc_signal:.2f} "
        f"membrane_residual="
        f"{membrane_residual_signal:.2f}"
    )

# ============================================================
# SAVE THE EXTENDED CELL SIGNAL TABLE
# ============================================================

# Convert all cell-result dictionaries into a DataFrame.
df_signal = pd.DataFrame(
    rows
)

# Save the extended cell-level signal table.
df_signal.to_csv(
    outpath(
        "cell_signal_vs_lysosome_signal_extended.csv"
    ),
    index=False
)

# Print the saved output path.
print(
    "[Signal quantification] Saved:",
    outpath(
        "cell_signal_vs_lysosome_signal_extended.csv"
    )
)

# ============================================================
# DIAMETER-INTERVAL SUBSET
#
# Create a subset used for visualization and Napari editing.
# ============================================================
# Copy the complete classified lysosome table.
df_interval = df.copy()


# Confirm that the interval object is a nonempty DataFrame.
if (
    isinstance(df_interval, pd.DataFrame)
    and len(df_interval) > 0
):
    # Apply the optional minimum-diameter filter.
    if DIAMETER_MIN_UM is not None:

        df_interval = df_interval[
            df_interval["diameter_um"]
            >= float(DIAMETER_MIN_UM)
        ].copy()

    # Apply the optional maximum-diameter filter.
    if DIAMETER_MAX_UM is not None:

        df_interval = df_interval[
            df_interval["diameter_um"]
            <= float(DIAMETER_MAX_UM)
        ].copy()

# Continue when filtered rows remain.
if (
    isinstance(df_interval, pd.DataFrame)
    and len(df_interval) > 0
):
    # Map original cell IDs to visualization IDs.
    df_interval["cell_id_ch2_viz"] = (
        df_interval[
            "cell_id_ch2"
        ]
        .map(cell_id_map_viz)
        .fillna(0)
        .astype(int)
    )

    # Initialize interval-specific per-cell lysosome IDs.
    df_interval["lys_id_in_cell"] = 0

    # Identify intracellular lysosomes with valid cell IDs.
    mask_in_i = (
        (
            df_interval.get(
                "location_ch2",
                ""
            )
            == "cell"
        )
        & (
            df_interval.get(
                "cell_id_ch2",
                0
            )
            > 0
        )
    )

    # Verify that the selection contains at least one row.
    if np.any(
        mask_in_i.to_numpy()
        if hasattr(
            mask_in_i,
            "to_numpy"
        )
        else mask_in_i
    ):

        # Sort intracellular interval lysosomes consistently.
        df_sorted_i = (
            df_interval.loc[
                mask_in_i
            ]
            .sort_values(
                [
                    "cell_id_ch2",
                    "z_um",
                    "y_um",
                    "x_um"
                ]
            )
            .copy()
        )

        # Assign serial IDs within each cell.
        df_interval.loc[
            df_sorted_i.index,
            "lys_id_in_cell"
        ] = (
            df_sorted_i
            .groupby(
                "cell_id_ch2"
            )
            .cumcount()
            .to_numpy()
            + 1
        ).astype(int)

# Save the diameter-filtered lysosome table.
df_interval.to_csv(
    outpath(
        "lysosomes_with_cell_vs_outside_diameter_interval.csv"
    ),
    index=False
)


# Print the interval-table path.
print(
    "Saved:",
    outpath(
        "lysosomes_with_cell_vs_outside_diameter_interval.csv"
    )
)

# ============================================================
# NAPARI EDITING DATASET
#
# Napari edits only the diameter-filtered subset.
# ============================================================
# Copy the interval subset for editing.
df_for_editing = df_interval.copy()


# Replace the working DataFrame with the editing subset.
df = df_for_editing


# Use the same interval subset for visualization.
df_viz = df_for_editing

# ============================================================
# FULL-SIZE OVERLAY EXPORT
# ============================================================
# Export cell labels and diameter-filtered lysosomes over
# the full-size raw image.
export_fullsize_overlay_stack(
    img_ch1=img_ch1,
    img_ch2_raw=img_ch2,
    cell_seg_viz=cell_seg_viz,
    df=df_viz,
    vx_um=vx_um,
    vy_um=vy_um,
    vz_um=vz_um,
    output_dir=output_dir,
    alpha_labels=0.45,
    draw_only_inside=True,
    fps=FPS,
    basename=(
        "FULLSIZE_overlay_CellID_"
        "Lysosomes_diameter_interval"
    )
)

# ============================================================
# VIDEO GENERATION
# ============================================================
# Generate videos only when enabled in the GUI.
if GENERATE_VIDEOS:

    # Convert normalized channel 2 into uint8.
    img_norm_2 = (
        ch2 * 255
    ).astype(np.uint8)

    # List for fused visualization frames.
    frames_fused = []

    # Read the number of slices.
    Z = img_norm_2.shape[0]

    # Calculate a representative XY pixel dimension.
    px_um_xy = float(
        np.sqrt(
            vx_um * vy_um
        )
    )

    # Process every slice.
    for z in range(Z):

        # Convert grayscale channel 2 into BGR format.
        base = cv2.cvtColor(
            img_norm_2[z],
            cv2.COLOR_GRAY2BGR
        )

        # Convert the visible cell mask to uint8.
        cell = (
            cell_mask_viz[z].astype(np.uint8)
            * 255
        )

        # Start the overlay as a copy of the base image.
        overlay = base.copy()

        # Increase the green channel where cells are present.
        overlay[
            ...,
            1
        ] = np.maximum(
            overlay[..., 1],
            cell
        )

        # Blend the green cell overlay with the base image.
        overlay = cv2.addWeighted(
            base,
            1.0,
            overlay,
            0.35,
            0.0
        )

        # Track whether any DataFrame-derived circles were drawn.
        drew_any = False


        # Check that the visualization table is valid and nonempty.
        if (
            isinstance(df_viz, pd.DataFrame)
            and len(df_viz) > 0
        ):

            # Keep only rows with finite coordinates and radii.
            dfv = df_viz[
                np.isfinite(df_viz["z_um"])
                & np.isfinite(df_viz["y_um"])
                & np.isfinite(df_viz["x_um"])
                & np.isfinite(df_viz["radius_um"])
            ]

            # Continue if valid rows remain.
            if not dfv.empty:

                # Convert Z positions to slice coordinates.
                zc = (
                    dfv["z_um"].to_numpy()
                    / vz_um
                ).astype(float)

                # Convert Y positions to pixel coordinates.
                yc = (
                    dfv["y_um"].to_numpy()
                    / vy_um
                ).astype(float)

                # Convert X positions to pixel coordinates.
                xc = (
                    dfv["x_um"].to_numpy()
                    / vx_um
                ).astype(float)

                # Extract physical radii.
                r_um = (
                    dfv["radius_um"]
                    .to_numpy()
                    .astype(float)
                )

                # Calculate Z displacement in voxels.
                dz_vox = np.abs(
                    zc - z
                )

                # Convert Z displacement to micrometers.
                dz_um = (
                    dz_vox * vz_um
                )

                # Select spheres intersecting this slice.
                hits = (
                    dz_um <= r_um
                )

                # Continue if at least one sphere intersects.
                if np.any(hits):

                    # Calculate projected circular radius.
                    r_proj_um = np.sqrt(
                        np.clip(
                            r_um[hits]**2
                            - dz_um[hits]**2,
                            0.0,
                            None
                        )
                    )

                    # Convert projected radius to XY pixels.
                    r_proj_vox = (
                        r_proj_um
                        / max(
                            px_um_xy,
                            1e-12
                        )
                    )

                    # Round Y centers to integer pixels.
                    ys = np.rint(
                        yc[hits]
                    ).astype(int)

                    # Round X centers to integer pixels.
                    xs = np.rint(
                        xc[hits]
                    ).astype(int)

                    # Read frame height and width.
                    H, W = (
                        cell_mask_viz.shape[1],
                        cell_mask_viz.shape[2]
                    )

                    # Set the smallest visible circle radius.
                    min_radius_px = 3

                    # Set the inner circle thickness.
                    thickness = 2

                    # Draw every projected lysosome circle.
                    for y, x, rpv in zip(
                        ys,
                        xs,
                        r_proj_vox
                    ):

                        # Round radius and enforce a minimum.
                        rr = int(
                            max(
                                min_radius_px,
                                round(rpv)
                            )
                        )

                        # Draw only valid visible circles.
                        if (
                            0 <= y < H
                            and 0 <= x < W
                            and rr > 0
                        ):

                            # Draw thick black outer outline.
                            cv2.circle(
                                overlay,
                                (x, y),
                                rr,
                                (0, 0, 0),
                                thickness + 2,
                                lineType=cv2.LINE_AA
                            )

                            # Draw cyan/yellow inner outline
                            # using OpenCV's BGR color ordering.
                            cv2.circle(
                                overlay,
                                (x, y),
                                rr,
                                (255, 255, 0),
                                thickness,
                                lineType=cv2.LINE_AA
                            )

                    # Record that DataFrame circles were drawn.
                    drew_any = True

        # Use the original blob array as a fallback when
        # no DataFrame-derived circles were drawn.
        if (
            not drew_any
            and blobs is not None
            and len(blobs) > 0
        ):

            # Keep blobs centered on the current Z slice.
            z_blobs = blobs[
                np.abs(
                    blobs[:, 0] - z
                ) < 0.5
            ]

            # Read frame dimensions.
            H, W = (
                cell_mask_viz.shape[1],
                cell_mask_viz.shape[2]
            )

            # Set minimum circle radius.
            min_radius_px = 3

            # Set line thickness.
            thickness = 2

            # Draw every fallback blob.
            for b in z_blobs:

                # Read rounded Y and X centers.
                y = int(
                    round(b[1])
                )
                x = int(
                    round(b[2])
                )

                # Read and clamp the radius.
                rpx = int(
                    max(
                        min_radius_px,
                        round(b[3])
                    )
                )

                # Draw only valid visible circles.
                if (
                    0 <= y < H
                    and 0 <= x < W
                    and rpx > 0
                ):

                    # Draw black outer outline.
                    cv2.circle(
                        overlay,
                        (x, y),
                        rpx,
                        (0, 0, 0),
                        thickness + 2,
                        lineType=cv2.LINE_AA
                    )

                    # Draw colored inner outline.
                    cv2.circle(
                        overlay,
                        (x, y),
                        rpx,
                        (255, 255, 0),
                        thickness,
                        lineType=cv2.LINE_AA
                    )


        # Add a descriptive label to the frame.
        cv2.putText(
            overlay,
            "FUSED (diameter interval + viz mask)",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Add the completed frame to the frame list.
        frames_fused.append(
            overlay
        )

    # Attempt to save the fused frames as MP4.
    try:

        imageio.mimsave(
            outpath(
                "ch2_fused_cell_diameter_interval.mp4"
            ),
            frames_fused,
            fps=FPS,
            format="FFMPEG"
        )

        print(
            "Saved:",
            outpath(
                "ch2_fused_cell_diameter_interval.mp4"
            )
        )

    except TypeError:

        # Save an animated GIF if the MP4 call raises TypeError.
        imageio.mimsave(
            outpath(
                "ch2_fused_cell_diameter_interval.gif"
            ),
            frames_fused,
            fps=FPS
        )

        print(
            "Saved:",
            outpath(
                "ch2_fused_cell_diameter_interval.gif"
            )
        )


    # Normalize raw channel 1.
    ch1_u8 = _norm_u8_stack(
        img_ch1.astype(np.float32)
    )

    # Rescale and normalize the smoothed channel 2.
    ch2_u8 = _norm_u8_stack(
        ch2.astype(np.float32)
        * 255.0
        / max(
            1.0,
            ch2.max()
        )
    )

    # Allocate raw-frame list.
    frames_raw = []

    # Allocate fused-frame list.
    frames_fused_all = []

    # Allocate side-by-side frame list.
    frames_side_by_side = []

    # Process every Z slice.
    for z in range(Z):

        # Assign channel 1 to blue.
        b = ch1_u8[
            z
        ]

        # Assign channel 2 to green.
        g = ch2_u8[
            z
        ]

        # Assign channel 1 to red.
        r = ch1_u8[
            z
        ]

        # Combine the three channels.
        base = np.dstack(
            [
                b,
                g,
                r
            ]
        )

        # Convert the visible cell mask to uint8.
        cell = (
            cell_mask_viz[z].astype(np.uint8)
            * 255
        )

        # Create overlay copy.
        overlay = base.copy()

        # Increase green values where cells are present.
        overlay[
            ...,
            1
        ] = np.maximum(
            overlay[..., 1],
            cell
        )

        # Blend overlay with base.
        overlay = cv2.addWeighted(
            base,
            1.0,
            overlay,
            0.35,
            0.0
        )

        # Track whether circles were drawn.
        drew_any = False


        # Check the visualization DataFrame.
        if (
            isinstance(df_viz, pd.DataFrame)
            and len(df_viz) > 0
        ):

            # Keep rows with finite measurements.
            dfv = df_viz[
                np.isfinite(df_viz["z_um"])
                & np.isfinite(df_viz["y_um"])
                & np.isfinite(df_viz["x_um"])
                & np.isfinite(df_viz["radius_um"])
            ]

            # Continue when valid rows remain.
            if not dfv.empty:

                # Convert Z coordinates to slices.
                zc = (
                    dfv["z_um"].to_numpy()
                    / vz_um
                ).astype(float)

                # Convert Y coordinates to pixels.
                yc = (
                    dfv["y_um"].to_numpy()
                    / vy_um
                ).astype(float)

                # Convert X coordinates to pixels.
                xc = (
                    dfv["x_um"].to_numpy()
                    / vx_um
                ).astype(float)

                # Extract radii in micrometers.
                r_um = (
                    dfv["radius_um"]
                    .to_numpy()
                    .astype(float)
                )

                # Calculate Z displacement in voxels.
                dz_vox = np.abs(
                    zc - z
                )

                # Convert displacement to micrometers.
                dz_um = (
                    dz_vox * vz_um
                )

                # Find lysosome spheres intersecting this slice.
                hits = (
                    dz_um <= r_um
                )

                # Continue when at least one intersection exists.
                if np.any(hits):

                    # Calculate projected radii.
                    r_proj_um = np.sqrt(
                        np.clip(
                            r_um[hits]**2
                            - dz_um[hits]**2,
                            0.0,
                            None
                        )
                    )

                    # Convert projected radii to pixels.
                    r_proj_vox = (
                        r_proj_um
                        / max(
                            px_um_xy,
                            1e-12
                        )
                    )

                    # Round Y centers.
                    ys = np.rint(
                        yc[hits]
                    ).astype(int)

                    # Round X centers.
                    xs = np.rint(
                        xc[hits]
                    ).astype(int)

                    # Read image dimensions.
                    H, W = (
                        cell_mask_viz.shape[1],
                        cell_mask_viz.shape[2]
                    )

                    # Set minimum display radius.
                    min_radius_px = 3

                    # Set line thickness.
                    thickness = 2

                    # Draw projected circles.
                    for y, x, rpv in zip(
                        ys,
                        xs,
                        r_proj_vox
                    ):

                        # Round and clamp radius.
                        rr = int(
                            max(
                                min_radius_px,
                                round(rpv)
                            )
                        )

                        # Draw only visible valid circles.
                        if (
                            0 <= y < H
                            and 0 <= x < W
                            and rr > 0
                        ):

                            # Draw black outer boundary.
                            cv2.circle(
                                overlay,
                                (x, y),
                                rr,
                                (0, 0, 0),
                                thickness + 2,
                                lineType=cv2.LINE_AA
                            )

                            # Draw colored inner boundary.
                            cv2.circle(
                                overlay,
                                (x, y),
                                rr,
                                (255, 255, 0),
                                thickness,
                                lineType=cv2.LINE_AA
                            )

                    # Mark the frame as successfully annotated.
                    drew_any = True

        # Fall back to original blob coordinates when necessary.
        if (
            not drew_any
            and blobs is not None
            and len(blobs) > 0
        ):

            # Select blobs centered on the current slice.
            z_blobs = blobs[
                np.abs(
                    blobs[:, 0] - z
                ) < 0.5
            ]

            # Read dimensions.
            H, W = (
                cell_mask_viz.shape[1],
                cell_mask_viz.shape[2]
            )

            # Set minimum radius.
            min_radius_px = 3

            # Set thickness.
            thickness = 2

            # Draw each fallback blob.
            for b_ in z_blobs:

                # Read Y and X centers.
                y = int(
                    round(b_[1])
                )
                x = int(
                    round(b_[2])
                )

                # Read radius.
                rpx = int(
                    max(
                        min_radius_px,
                        round(b_[3])
                    )
                )

                # Draw valid circles.
                if (
                    0 <= y < H
                    and 0 <= x < W
                    and rpx > 0
                ):

                    # Draw outer black line.
                    cv2.circle(
                        overlay,
                        (x, y),
                        rpx,
                        (0, 0, 0),
                        thickness + 2,
                        lineType=cv2.LINE_AA
                    )

                    # Draw inner colored line.
                    cv2.circle(
                        overlay,
                        (x, y),
                        rpx,
                        (255, 255, 0),
                        thickness,
                        lineType=cv2.LINE_AA
                    )

        # Label the raw frame.
        cv2.putText(
            base,
            "RAW (Ch1+Ch2)",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # Label the fused frame.
        cv2.putText(
            overlay,
            "FUSED (diameter interval + viz mask)",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Save the raw frame.
        frames_raw.append(
            base
        )

        # Save the fused frame.
        frames_fused_all.append(
            overlay
        )

        # Concatenate raw and fused frames horizontally.
        frames_side_by_side.append(
            cv2.hconcat(
                [
                    base,
                    overlay
                ]
            )
        )

    def _save_video_sync(
        basename,
        frames
    ):
        """
        Save frames as MP4, with GIF fallback.
        """

        try:

            # Open an FFMPEG MP4 writer.
            with imageio.get_writer(
                outpath(
                    f"{basename}.mp4"
                ),
                fps=FPS,
                format="FFMPEG",
                codec="libx264",
                macro_block_size=None
            ) as w:

                # Append every frame.
                for fr in frames:
                    w.append_data(
                        fr
                    )

            # Print the saved MP4 path.
            print(
                f"Saved: "
                f"{outpath(f'{basename}.mp4')} "
                f"@ {FPS} fps"
            )

        except Exception:

            # Save GIF when MP4 creation fails.
            imageio.mimsave(
                outpath(
                    f"{basename}.gif"
                ),
                frames,
                duration=1.0 / FPS
            )

            # Print the saved GIF path.
            print(
                f"Saved: "
                f"{outpath(f'{basename}.gif')} "
                f"@ {FPS} fps equivalent"
            )


    # Save fused-only visualization.
    _save_video_sync(
        "ch2_fused_all_viz_diameter_interval",
        frames_fused_all
    )

    # Save raw visualization.
    _save_video_sync(
        "ch2_raw",
        frames_raw
    )

    # Save raw and fused views side by side.
    _save_video_sync(
        "ch2_raw_and_fused_all_viz_diameter_interval",
        frames_side_by_side
    )

# ============================================================
# PER-LYSOSOME SIGNAL
# ============================================================
# Announce per-lysosome quantification.
print(
    "[Per-lysosome signal] Computing..."
)

# Convert channel 1 to float.
signal_img = img_ch1.astype(
    np.float32
)

# Estimate channel-1 background.
bg = np.percentile(
    signal_img,
    5
)

# Subtract background and clip negative values.
signal_img = np.clip(
    signal_img - bg,
    0,
    None
)

# List for one total signal measurement per lysosome.
lys_signal_individual = []

# Process every row in the current DataFrame.
#
# At this point, df contains the diameter-filtered subset.
for _, row in df.iterrows():

    # Generate a sphere for the current lysosome.
    (
        z1,
        z2,
        y1,
        y2,
        x1,
        x2,
        sphere
    ) = sphere_mask(
        row["z_um"],
        row["y_um"],
        row["x_um"],
        row["radius_um"]
    )

    # Extract the local corrected signal patch.
    patch = signal_img[
        z1:z2,
        y1:y2,
        x1:x2
    ]

    # Sum signal only inside the sphere.
    val = float(
        patch[
            sphere
        ].sum()
    )

    # Store the per-lysosome signal.
    lys_signal_individual.append(
        val
    )

# Add per-lysosome signal to the DataFrame.
df["lysosome_signal_individual"] = np.array(
    lys_signal_individual,
    dtype=float
)

# Confirm completion.
print(
    "[Per-lysosome signal] Done."
)

# ============================================================
# SLICE-BY-SLICE OVERLAP VISUALIZATION
#
# Grayscale = raw channel 1
# Blue      = channel-2-derived mask
# Red/yellow = channel-1 signal overlapping the mask
# ============================================================
# Convert the quantification mask to Boolean.
mask_5vox = neuron_mask_for_quantification.astype(
    bool
)

# Convert raw channel 1 to float.
ch1_raw = img_ch1.astype(
    np.float32
)

# Normalize channel 1 for visualization.
ch1_u8 = _norm_u8_stack(
    ch1_raw
)

# Estimate the channel-1 background threshold.
ch1_bg = np.percentile(
    ch1_raw,
    5
)

# Mark voxels whose raw signal is above the background threshold.
ch1_signal_mask = (
    ch1_raw > ch1_bg
)

# Select channel-1 signal voxels inside the mask.
overlap_mask = (
    mask_5vox
    & ch1_signal_mask
)

# Read normalized image dimensions.
Z, H, W = ch1_u8.shape

# Allocate the RGB output stack.
frames = np.zeros(
    (
        Z,
        H,
        W,
        3
    ),
    dtype=np.uint8
)

# Process every slice.
for z in range(Z):

    # Create a grayscale RGB base image.
    base = np.dstack(
        [
            ch1_u8[z],
            ch1_u8[z],
            ch1_u8[z]
        ]
    ).astype(np.float32)

    # Create a pure-blue overlay image.
    blue = np.zeros_like(
        base
    )

    blue[
        ...,
        2
    ] = 255


    # Create a red/orange overlap image.
    overlap = np.zeros_like(
        base
    )

    overlap[
        ...,
        0
    ] = 255

    overlap[
        ...,
        1
    ] = 80


    # Extract the mask for the current slice.
    mask_z = mask_5vox[
        z
    ]

    # Extract overlap pixels for the current slice.
    overlap_z = overlap_mask[
        z
    ]


    # Start the result as a copy of the grayscale base.
    out = base.copy()


    # Blend blue over mask pixels.
    out[
        mask_z
    ] = (
        0.55 * out[mask_z]
        + 0.45 * blue[mask_z]
    )


    # Blend red/orange strongly over overlap pixels.
    out[
        overlap_z
    ] = (
        0.25 * out[overlap_z]
        + 0.75 * overlap[overlap_z]
    )


    # Clip and save the RGB slice.
    frames[z] = np.clip(
        out,
        0,
        255
    ).astype(np.uint8)


# Save the slice-by-slice overlap TIFF.
tiff.imwrite(
    outpath(
        "SLICE_BY_SLICE_Ch1_overlap_with_5vox_Ch2_mask.tif"
    ),
    frames,
    photometric="rgb"
)

# Print the saved path.
print(
    "Saved:",
    outpath(
        "SLICE_BY_SLICE_Ch1_overlap_with_5vox_Ch2_mask.tif"
    )
)

# ============================================================
# PERCENTAGE OF CHANNEL-1 SIGNAL INSIDE MASK PER CELL
# ============================================================
# Announce this quantification stage.
print(
    "\n[Signal in mask per cell] Starting..."
)

# Convert raw channel 1 to float.
signal_img = img_ch1.astype(
    np.float32
)

# Estimate background from the fifth percentile.
bg = np.percentile(
    signal_img,
    5
)

# Subtract background and clip negative values.
signal_img = np.clip(
    signal_img - bg,
    0,
    None
)

# Print the estimated background.
print(
    f"[DEBUG] Background CH1 = {bg:.3f}"
)

# Extract all unique watershed cell labels.
cell_ids = np.unique(
    cell_seg
)

# Remove the background label.
cell_ids = cell_ids[
    cell_ids > 0
]

# Initialize per-cell result rows.
rows = []

# Process every cell.
for cid in cell_ids:
    # Select this cell within the quantification mask.
    cell_mask = (
        (cell_seg == cid)
        & neuron_mask_for_quantification
    )

    # Intersect the quantified cell region with the
    # original neuron mask.
    mask_in_cell = (
        cell_mask
        & neuron_mask
    )

    # Calculate total corrected channel-1 signal
    # in the quantified cell region.
    total_cell_signal = float(
        signal_img[
            cell_mask
        ].sum()
    )

    # Calculate corrected channel-1 signal
    # inside the intersected mask.
    mask_signal = float(
        signal_img[
            mask_in_cell
        ].sum()
    )

    # Calculate the percentage of total cell signal
    # located inside mask_in_cell.
    signal_percent_in_mask = (
        mask_signal
        / total_cell_signal
        * 100.0
    ) if total_cell_signal > 0 else 0.0

    # Add this cell's measurements to the output list.
    rows.append({
        "cell_id":
            int(cid),

        "cell_signal_total_CH1":
            total_cell_signal,

        "signal_inside_mask_CH1":
            mask_signal,

        "signal_percent_inside_mask":
            signal_percent_in_mask,
    })

    # Print this cell's results.
    print(
        f"[CELL {cid}] "
        f"total_CH1={total_cell_signal:.2f} | "
        f"mask_CH1={mask_signal:.2f} | "
        f"percent={signal_percent_in_mask:.2f}%"
    )

# Convert per-cell result dictionaries into a DataFrame.
df_signal_mask = pd.DataFrame(
    rows
)

# Save the per-cell signal-percentage table.
df_signal_mask.to_csv(
    outpath(
        "cell_CH1_signal_percent_inside_mask.csv"
    ),
    index=False
)

# Print the saved path.
print(
    "[Signal in mask per cell] Saved:",
    outpath(
        "cell_CH1_signal_percent_inside_mask.csv"
    )
)

# Calculate total corrected channel-1 signal
# inside all watershed cells.
total_signal_all = signal_img[
    cell_seg > 0
].sum()

# Calculate corrected channel-1 signal inside voxels that
# belong to both a watershed cell and the original neuron mask.
mask_signal_all = signal_img[
    (cell_seg > 0)
    & neuron_mask
].sum()

# Calculate the global percentage of cellular signal
# inside the original neuron mask.
global_percent = (
    mask_signal_all
    / total_signal_all
    * 100
)

# Print the global percentage.
print(
    f"Global CH1 signal in mask = "
    f"{global_percent:.2f}%"
)

# ======================================================================
# NAPARI BLOCK
#
# This section opens an interactive Napari viewer and adds:
#
#   1. Raw channel 1 and channel 2 images.
#   2. Lysosome core and associated-region fills.
#   3. Lysosome core and associated-region outlines.
#   4. The membrane-minus-lysosomes intensity image.
#   5. Keyboard shortcuts for switching between 2D and 3D.
# ======================================================================

# Check whether the GUI configuration requested
# automatic launch of the Napari viewer.
if LAUNCH_VIEWER:

    # Create a new Napari viewer window.
    viewer = napari.Viewer()

    # Start the viewer in three-dimensional display mode.
    #
    # ndisplay = 3:
    #   display the image volume in 3D.
    #
    # ndisplay = 2:
    #   display one slice at a time.
    viewer.dims.ndisplay = 3

    # Add the raw second image channel to the viewer.
    #
    # This channel represents the cell or membrane signal.
    viewer.add_image(
        img_ch2,
        name="Ch2 raw"
    )

    # Add the raw first image channel to the viewer.
    #
    # This channel represents the lysosome signal.
    viewer.add_image(
        img_ch1,
        name="Ch1 raw"
    )

    # ========================================================
    # 2D LYSOSOME FILLS AND OUTLINES OVER RAW DATA
    #
    # The following masks are calculated independently
    # for every XY slice.
    #
    # Two regions are shown:
    #
    #   Core:
    #       The directly detected lysosome sphere.
    #
    #   Associated:
    #       The lysosome core plus the surrounding halo.
    # ========================================================

    # --------------------------------------------------------
    # CREATE BINARY FILLS
    # --------------------------------------------------------
    # Convert the Boolean lysosome-core mask into uint8.
    #
    # False becomes 0.
    # True becomes 1.
    core_fill = lys_mask.astype(
        np.uint8
    )

    # Convert the Boolean lysosome-associated mask into uint8.
    assoc_fill = lys_assoc_mask.astype(
        np.uint8
    )

    # --------------------------------------------------------
    # OUTLINE THICKNESS
    # --------------------------------------------------------
    # Set the core-outline thickness to three XY pixels.
    #
    # The core outline will therefore appear relatively thick.
    CORE_OUTLINE_THICKNESS = 3

    # Set the associated-region outline thickness
    # to one XY pixel.
    ASSOC_OUTLINE_THICKNESS = 1

    # Allocate an empty uint8 volume for the core outline.
    #
    # It has the same Z, Y, and X dimensions as core_fill.
    core_outline = np.zeros_like(
        core_fill,
        dtype=np.uint8
    )

    # Allocate an empty uint8 volume for the
    # associated-region outline.
    assoc_outline = np.zeros_like(
        assoc_fill,
        dtype=np.uint8
    )

    # Create a two-dimensional disk-shaped structuring element
    # for the thick core outline.
    core_disk = disk(
        CORE_OUTLINE_THICKNESS
    )

    # Create a two-dimensional disk-shaped structuring element
    # for the thin associated-region outline.
    assoc_disk = disk(
        ASSOC_OUTLINE_THICKNESS
    )

    # Process each Z slice independently.
    for z in range(
        core_fill.shape[0]
    ):
        # ----------------------------------------------------
        # THICK CORE OUTLINE
        # ----------------------------------------------------
        # Construct the core boundary using:
        #
        # original mask AND NOT eroded mask.
        #
        # Pixels removed by erosion form the outline.
        core_outline[z] = (
            core_fill[z]

            # Remove the eroded interior from the original fill.
            & ~binary_erosion(
                core_fill[z],

                # Use the thick two-dimensional disk footprint.
                footprint=core_disk
            )
        )

        # ----------------------------------------------------
        # THIN ASSOCIATED-REGION OUTLINE
        # ----------------------------------------------------
        # Construct the associated-region boundary using
        # the same original-minus-eroded approach.
        assoc_outline[z] = (
            assoc_fill[z]

            # Remove the eroded interior.
            & ~binary_erosion(
                assoc_fill[z],

                # Use the thinner disk footprint.
                footprint=assoc_disk
            )
        )

    # ========================================================
    # ASSOCIATED-REGION FILL
    # ========================================================
    # Add the associated-region fill as a Napari Labels layer.
    assoc_fill_layer = viewer.add_labels(
        assoc_fill,

        # Name displayed in the Napari layer list.
        name="Lysosome ASSOCIATED fill",

        # Use very low opacity so the raw image remains visible.
        opacity=0.08,

        # Make the layer visible when the viewer opens.
        visible=True,
    )

    # Assign custom colors to the associated-fill labels.
    assoc_fill_layer.color = {

        # Label 0 is transparent background.
        0: (
            0,
            0,
            0,
            0
        ),

        # Label 1 is opaque white.
        #
        # Layer opacity still reduces its displayed strength.
        1: (
            1.00,
            1.00,
            1.00,
            1.00
        )
    }

    # ========================================================
    # ASSOCIATED-REGION OUTLINE
    # ========================================================
    # Add the thin associated-region outline as a Labels layer.
    assoc_outline_layer = viewer.add_labels(
        assoc_outline,

        # Layer name shown in Napari.
        name="Lysosome ASSOCIATED outline",

        # Use moderately high opacity.
        opacity=0.6,

        # Display the outline initially.
        visible=True,
    )

    # Assign colors to the associated-outline labels.
    assoc_outline_layer.color = {

        # Transparent background.
        0: (
            0,
            0,
            0,
            0
        ),

        # White outline.
        1: (
            1.00,
            1.00,
            1.00,
            1.00
        )
    }

    # ========================================================
    # CORE FILL
    # ========================================================
    # Add the lysosome-core fill as a Labels layer.
    core_fill_layer = viewer.add_labels(
        core_fill,

        # Name displayed in the Napari layer list.
        name="Lysosome CORE fill",

        # Use partial opacity.
        opacity=0.3,

        # Make the layer visible initially.
        visible=True,
    )

    # Assign colors to the core-fill labels.
    core_fill_layer.color = {

        # Transparent background.
        0: (
            0,
            0,
            0,
            0
        ),

        # Black lysosome-core fill.
        1: (
            0.00,
            0.00,
            0.00,
            1.00
        )
    }

    # ========================================================
    # CORE OUTLINE
    # ========================================================
    # Add the thick lysosome-core outline as a Labels layer.
    core_outline_layer = viewer.add_labels(
        core_outline,

        # Name displayed in the layer list.
        name="Lysosome CORE outline",

        # Display the outline at full opacity.
        opacity=1.0,

        # Make it visible initially.
        visible=True,
    )

    # Assign colors to the core-outline labels.
    core_outline_layer.color = {

        # Transparent background.
        0: (
            0,
            0,
            0,
            0
        ),

        # Black core outline.
        1: (
            0.00,
            0.00,
            0.00,
            1.00
        )
    }

    # ========================================================
    # IMPROVED LABEL-LAYER RENDERING
    # ========================================================
    # Process all four lysosome label layers.
    for layer in (
        assoc_fill_layer,
        assoc_outline_layer,
        core_fill_layer,
        core_outline_layer,
    ):

        try:

            # Use translucent rendering without depth testing.
            #
            # This can make overlapping transparent layers
            # easier to see in 3D.
            layer.blending = (
                "translucent_no_depth"
            )

        except Exception:

            # Ignore errors if the installed Napari version
            # does not support this blending mode.
            pass

    # ==============================================================
    # MEMBRANE AND LYSOSOME VISUALIZATION LAYERS
    # ==============================================================
    # Define the physical scale of the image volume.
    #
    # Napari expects scale values in array-axis order:
    #
    #   Z, Y, X
    image_scale = (
        float(vz_um),
        float(vy_um),
        float(vx_um)
    )

    # Raw channel 1 is not added again here because
    # "Ch1 raw" was already added above.

    # Add the final residual membrane intensity image.
    #
    # This image contains channel-1 signal in:
    #
    # membrane mask minus the lysosome-associated region.
    viewer.add_image(
        ch1_membrane_residual,

        # Name displayed in the Napari layer list.
        name=" Ch 03 - membrane minus lysosomes",

        # Display this channel using magenta.
        colormap="magenta",

        # Add intensities from this layer to other image layers.
        blending="additive",

        # Physical scaling could be enabled using:
        #
        # scale=image_scale,
        #
        # It is currently commented out in the original code.
        # scale=image_scale,

        # Display at full opacity.
        opacity=1.0,

        # Use maximum-intensity projection in 3D mode.
        rendering="mip",

        # Make the layer visible initially.
        visible=True
    )

    # --------------------------------------------------------------
    # KEYBOARD CONTROLS
    #
    # Press 2:
    #   Switch to slice-by-slice two-dimensional viewing.
    #
    # Press 3:
    #   Switch to three-dimensional viewing.
    # --------------------------------------------------------------

    # Bind the keyboard key "2" to the following function.
    #
    # overwrite=True replaces any existing Napari binding
    # assigned to the same key.
    @viewer.bind_key(
        "2",
        overwrite=True
    )
    def show_result_in_2d(viewer):
        """
        Switch the Napari viewer to two-dimensional slice mode.
        """
        # Display one Z slice at a time.
        viewer.dims.ndisplay = 2

        # Print instructions for navigating the Z stack.
        print(
            "[Napari] 2D slice mode. "
            "Move the Z slider to inspect individual slices."
        )

    # Bind the keyboard key "3" to the following function.
    @viewer.bind_key(
        "3",
        overwrite=True
    )
    def show_result_in_3d(viewer):
        """
        Switch the Napari viewer to three-dimensional mode.
        """

        # Display the image as a 3D volume.
        viewer.dims.ndisplay = 3

        # Explain that intensity images use MIP rendering.
        print(
            "[Napari] 3D mode. "
            "The intensity layers use maximum-intensity projection."
        )

    # ========================================================
    # CELL MASK LAYER
    # ========================================================

    # Add the Boolean visualization cell mask as a Napari
    # Labels layer.
    mask_layer = viewer.add_labels(

        # Convert the Boolean mask into unsigned 8-bit labels.
        #
        # Background becomes 0.
        # Foreground becomes 1.
        cell_mask_viz.astype(np.uint8),

        # Use a different layer name depending on the selected mode.
        name=(
            "Neurite mask"
            if NEURITE_MODE
            else "Cell mask"
        ),

        # Display the mask with partial transparency.
        opacity=0.35
    )


    # Attempt to improve transparent rendering.
    try:

        # Render transparent labels without depth testing.
        mask_layer.blending = "translucent_no_depth"

    except Exception:

        # Ignore the error when the installed Napari version
        # does not support this blending mode.
        pass


    # ========================================================
    # CELL LABEL COLOR MAP
    # ========================================================

    # Determine the number of visible cell labels.
    #
    # If cell_seg_viz is a NumPy array, the largest label
    # represents the highest serial ID.
    #
    # Otherwise, assume that there are no labels.
    n_labels = (
        int(cell_seg_viz.max())
        if isinstance(cell_seg_viz, np.ndarray)
        else 0
    )


    # Generate an RGB color for every serial cell label.
    #
    # The returned array contains uint8 RGB values
    # in the range 0–255.
    cmap_u8 = make_label_colormap(
        n_labels,
        seed_hue=0.13
    )


    # Initialize the Napari label-color dictionary.
    #
    # Label 0 represents the background and is fully transparent.
    label_color = {
        0: (
            0.0,
            0.0,
            0.0,
            0.0
        )
    }


    # Create one normalized RGBA color for every cell ID.
    for i in range(
        1,
        n_labels + 1
    ):

        # Read the uint8 RGB values assigned to this label.
        r, g, b = (
            cmap_u8[i].astype(np.float32)
            / 255.0
        )

        # Store the color as an RGBA tuple.
        #
        # Alpha is set to 1.0, meaning fully opaque before
        # the layer-level opacity is applied.
        label_color[i] = (
            float(r),
            float(g),
            float(b),
            1.0
        )


    # ========================================================
    # HELPER: ADD LABELS WITH CUSTOM COLORS
    # ========================================================

    def _add_labels_with_color(
        data,
        name,
        opacity=0.25,
        visible=True
    ):
        """
        Add a Napari Labels layer with the serial cell colors.

        The helper supports different Napari versions.

        Newer versions may accept the color dictionary directly
        in viewer.add_labels().

        Older versions may require assigning layer.color
        after creating the layer.
        """

        try:

            # Try adding the layer while passing the color map
            # directly to viewer.add_labels().
            layer = viewer.add_labels(
                data,

                # Name shown in the Napari layer list.
                name=name,

                # Convert opacity to a normal Python float.
                opacity=float(opacity),

                # Convert visibility to a Boolean.
                visible=bool(visible),

                # Assign the custom label-color dictionary.
                color=label_color
            )

            # Return the newly created layer.
            return layer

        except TypeError:

            # Fall back to adding the layer without the color argument.
            #
            # This branch supports Napari versions whose add_labels()
            # function does not accept color during construction.
            layer = viewer.add_labels(
                data,
                name=name,
                opacity=float(opacity),
                visible=bool(visible)
            )

            try:

                # Assign the color map after layer creation.
                layer.color = label_color

            except Exception:

                # Ignore color-assignment failures.
                pass

            # Return the fallback layer.
            return layer


    # ========================================================
    # SERIAL CELL-ID LAYER
    # ========================================================

    # Add the complete serial cell-label image.
    id_layer = _add_labels_with_color(

        # Convert labels to uint16.
        cell_seg_viz.astype(np.uint16),

        # Layer name.
        "ID (serial)",

        # Use relatively low opacity.
        opacity=0.25,

        # Display the layer initially.
        visible=True
    )


    # Attempt to improve transparent 3D rendering.
    try:
        id_layer.blending = "translucent_no_depth"

    except Exception:
        pass


    # ========================================================
    # FILTERED CELL-ID LAYER
    #
    # This layer is initially empty.
    #
    # It is later populated with only the IDs selected
    # through the filter controls.
    # ========================================================

    id_filtered_layer = _add_labels_with_color(

        # Begin with an all-zero label image.
        np.zeros_like(
            cell_seg_viz,
            dtype=np.uint16
        ),

        # Layer name displayed in Napari.
        "ID (filtered view)",

        # Use greater opacity than the complete ID layer.
        opacity=0.45,

        # Keep the layer hidden until a filter is applied.
        visible=False
    )


    # Attempt to apply the same transparent rendering mode.
    try:
        id_filtered_layer.blending = "translucent_no_depth"

    except Exception:
        pass


    # ========================================================
    # POINT-LAYER PLACEHOLDERS
    # ========================================================

    # This variable will later store the editable lysosome
    # Points layer.
    pts_layer = None


    # This variable will later store a filtered, view-only
    # lysosome Points layer.
    view_pts_layer = None


    # ========================================================
    # SERIAL-TO-ORIGINAL CELL-ID MAP
    # ========================================================

    # Initialize the reverse ID map as unavailable.
    serial_to_original_id = None


    # Verify that the original-to-serial map exists and is nonempty.
    if (
        isinstance(cell_id_map_viz, dict)
        and len(cell_id_map_viz) > 0
    ):

        # Reverse the mapping:
        #
        # original watershed ID -> serial ID
        #
        # becomes:
        #
        # serial ID -> original watershed ID
        serial_to_original_id = {
            serial_id: original_id
            for (
                original_id,
                serial_id
            ) in cell_id_map_viz.items()
        }


    # ========================================================
    # HELPER: CALCULATE POINT COLORS FROM PROPERTIES
    # ========================================================

    def _points_rgba_from_props(props_dict):
        """
        Generate one RGBA color for each lysosome point.

        Intracellular lysosomes receive the color assigned
        to their serial cell ID.

        Outside or unassigned lysosomes receive magenta.
        """

        # Read the cell/outside classification property.
        #
        # Missing property values become an empty array.
        loc = np.array(
            props_dict.get(
                "location_ch2",
                []
            ),
            dtype=str
        )


        # Read the serial cell-ID property.
        cid = np.array(
            props_dict.get(
                "cell_id_serial",
                []
            ),
            dtype=int
        )


        # Determine the number of points.
        n = int(
            len(cid)
        )


        # Allocate one RGBA row per point.
        rgba = np.zeros(
            (
                n,
                4
            ),
            dtype=np.float32
        )


        # Set every point to magenta by default.
        #
        # The alpha value is 0.85.
        rgba[:] = (
            1.0,
            0.0,
            1.0,
            0.85
        )


        # Identify points that:
        #
        #   1. are classified as inside a cell, and
        #   2. have a valid positive serial cell ID.
        inside = (
            (loc == "cell")
            & (cid > 0)
        )


        # Continue when at least one intracellular point exists.
        if np.any(inside):

            # Process each intracellular point index.
            for k in np.where(
                inside
            )[0]:

                # Read this point's serial cell ID.
                c = int(
                    cid[k]
                )

                # Look up the corresponding cell-label color.
                #
                # If the serial ID is absent from label_color,
                # use magenta as a fallback.
                rgba[k] = np.array(
                    label_color.get(
                        c,
                        (
                            1.0,
                            0.0,
                            1.0,
                            0.85
                        )
                    ),
                    dtype=np.float32
                )


        # Return the complete point-color array.
        return rgba


    # ========================================================
    # HELPER: REFRESH POINT-LAYER COLORS
    # ========================================================

    def _refresh_point_colors(layer):
        """
        Recalculate point colors from the layer properties.
        """

        # Do nothing when no layer was supplied.
        if layer is None:
            return


        # Copy the current point-property dictionary.
        p = dict(
            layer.properties
        )


        try:

            # Recalculate one face color for every point.
            layer.face_color = _points_rgba_from_props(
                p
            )

        except Exception:

            # Ignore errors caused by unsupported Napari versions
            # or inconsistent property lengths.
            pass


    # ========================================================
    # FILTER STATE
    #
    # This dictionary stores the currently active point
    # and label filtering configuration.
    # ========================================================

    _filter_state = {

        # Selected serial cell IDs.
        #
        # None means that no filter is currently active.
        "cell_ids": None,

        # Location filter:
        #   "both"
        #   "cell"
        #   "outside"
        "location": "both",

        # Selected per-cell lysosome serial IDs.
        #
        # None means all lysosome IDs.
        "lys_ids": None,

        # Whether the filtered cell-label layer should be shown.
        "show_filtered_labels": True,

        # Whether the filtered points layer should be shown.
        "show_view_layer": True,
    }


    # ========================================================
    # HELPER: PARSE CELL-ID TEXT
    # ========================================================

    def _parse_id_text(
        s,
        max_id
    ):
        """
        Parse cell-ID text entered by the user.

        Accepted examples
        -----------------
        all
        *
        1
        1,2,5
        1 2 5
        1;2;5
        3-8
        1,3-5,9

        Returns
        -------
        list
            Sorted valid serial cell IDs.
        """

        # Replace None with an empty string,
        # remove surrounding spaces, and convert to lowercase.
        s = (
            s or ""
        ).strip().lower()


        # Treat an empty entry, "all", or "*" as every valid ID.
        if s in (
            "",
            "all",
            "*"
        ):

            # Return consecutive IDs from 1 through max_id.
            return list(
                range(
                    1,
                    int(max_id) + 1
                )
            )


        # Split the input at commas, whitespace, or semicolons.
        parts = re.split(
            r"[,\s;]+",
            s
        )


        # Use a set to avoid duplicate IDs.
        out = set()


        # Process each text fragment.
        for part in parts:

            # Remove surrounding whitespace.
            part = part.strip()


            # Ignore empty fragments.
            if not part:
                continue


            # Check whether this fragment describes a range.
            if "-" in part:

                # Split at the first hyphen.
                a, b = part.split(
                    "-",
                    1
                )

                try:

                    # Convert the first range endpoint to integer.
                    a = int(a)

                    # Convert the second range endpoint to integer.
                    b = int(b)

                except Exception:

                    # Ignore malformed ranges.
                    continue


                # Arrange the endpoints in increasing order.
                lo, hi = (
                    (a, b)
                    if a <= b
                    else (b, a)
                )


                # Add every integer in the inclusive range.
                for v in range(
                    lo,
                    hi + 1
                ):

                    # Keep only IDs inside the valid interval.
                    if (
                        1
                        <= v
                        <= int(max_id)
                    ):
                        out.add(
                            v
                        )

            else:

                # This fragment should contain one integer ID.
                try:
                    v = int(
                        part
                    )

                except Exception:

                    # Ignore malformed values.
                    continue


                # Keep only valid positive serial IDs.
                if (
                    1
                    <= v
                    <= int(max_id)
                ):
                    out.add(
                        v
                    )


        # Return the selected IDs in ascending order.
        return sorted(
            out
        )


    # ========================================================
    # HELPER: PARSE LYSOSOME-ID TEXT
    # ========================================================

    def _parse_lys_text(s):
        """
        Parse per-cell lysosome serial IDs.

        The syntax is similar to _parse_id_text(), but there
        is no maximum-ID restriction.

        Returns
        -------
        list or None
            Sorted positive IDs, or None to represent all IDs.
        """

        # Normalize the text input.
        s = (
            s or ""
        ).strip().lower()


        # Empty text, "all", and "*" mean no lysosome-ID filter.
        if s in (
            "",
            "all",
            "*"
        ):
            return None


        # Split at commas, whitespace, or semicolons.
        parts = re.split(
            r"[,\s;]+",
            s
        )


        # Use a set to avoid duplicate values.
        out = set()


        # Process each fragment.
        for part in parts:

            # Remove surrounding spaces.
            part = part.strip()


            # Ignore empty fragments.
            if not part:
                continue


            # Handle an inclusive range.
            if "-" in part:

                # Split into range endpoints.
                a, b = part.split(
                    "-",
                    1
                )

                try:
                    a = int(a)
                    b = int(b)

                except Exception:

                    # Ignore malformed ranges.
                    continue


                # Arrange endpoints in increasing order.
                lo, hi = (
                    (a, b)
                    if a <= b
                    else (b, a)
                )


                # Add every positive integer in the range.
                for v in range(
                    lo,
                    hi + 1
                ):

                    if v >= 1:
                        out.add(
                            v
                        )

            else:

                # Handle one individual lysosome ID.
                try:
                    v = int(
                        part
                    )

                except Exception:
                    continue


                # Retain only positive IDs.
                if v >= 1:
                    out.add(
                        v
                    )


        # Return sorted IDs when any were parsed.
        #
        # Otherwise return None, meaning all IDs.
        return (
            sorted(out)
            if len(out)
            else None
        )


    # ========================================================
    # HELPER: UPDATE FILTERED CELL LABELS
    # ========================================================

    def _update_filtered_labels(sel_ids):
        """
        Update the filtered cell-ID layer.

        Only the selected serial cell IDs remain visible.
        """

        # Hide and clear the layer when no IDs are supplied.
        if (
            sel_ids is None
            or len(sel_ids) == 0
        ):

            # Hide the filtered layer.
            id_filtered_layer.visible = False

            # Replace its data with an all-zero label image.
            id_filtered_layer.data = np.zeros_like(
                cell_seg_viz,
                dtype=np.uint16
            )

            # Stop processing.
            return


        # Create a Boolean mask identifying voxels whose
        # serial label appears in sel_ids.
        mask = np.isin(
            cell_seg_viz,
            np.array(
                sel_ids,
                dtype=np.int32
            )
        )


        # Allocate an all-zero output label image.
        out = np.zeros_like(
            cell_seg_viz,
            dtype=np.uint16
        )


        # Copy the selected serial labels into the output.
        out[
            mask
        ] = cell_seg_viz[
            mask
        ].astype(np.uint16)


        # Replace the filtered layer's data.
        id_filtered_layer.data = out


        # Make the filtered label layer visible.
        id_filtered_layer.visible = True


    # ========================================================
    # HELPER: APPLY FILTER TO LYSOSOME POINTS
    # ========================================================

    def _apply_points_filter(
        sel_ids,
        location_mode="both",
        lys_ids=None,
        show_view_layer=True
    ):
        """
        Copy points matching the selected filters from the
        editable point layer into the view-only point layer.
        """

        # The operation requires both point layers.
        if (
            pts_layer is None
            or view_pts_layer is None
        ):
            return


        # Copy the editable layer's property dictionary.
        p = dict(
            pts_layer.properties
        )


        # Read the cell/outside classification.
        loc = np.array(
            p.get(
                "location_ch2",
                []
            ),
            dtype=str
        )


        # Read serial cell IDs.
        cid = np.array(
            p.get(
                "cell_id_serial",
                []
            ),
            dtype=int
        )


        # Read per-cell serial lysosome IDs.
        lys = np.array(
            p.get(
                "lys_id_serial",
                []
            ),
            dtype=int
        )


        # Convert selected cell IDs to a set of integers.
        #
        # An empty or None input becomes an empty set.
        sel_ids = set(
            int(x)
            for x in (
                sel_ids or []
            )
        )


        # Initially keep points whose serial cell ID
        # belongs to the selected cell-ID set.
        keep = np.isin(
            cid,
            list(sel_ids)
        )


        # Apply the intracellular-only filter.
        if location_mode == "cell":

            # Keep only points classified as "cell".
            keep &= (
                loc == "cell"
            )


        # Apply the outside-only filter.
        elif location_mode == "outside":

            # Keep points whose location is not "cell".
            keep &= (
                loc != "cell"
            )


        # Apply a lysosome serial-ID filter when supplied.
        if lys_ids is not None:

            # Convert selected lysosome IDs to integers.
            lys_ids = set(
                int(x)
                for x in lys_ids
            )

            # Keep only matching per-cell lysosome IDs.
            keep &= np.isin(
                lys,
                list(lys_ids)
            )


        # Convert the Boolean selection into point indices.
        idx = np.where(
            keep
        )[0]


        # Copy selected point coordinates into the view layer.
        view_pts_layer.data = np.asarray(
            pts_layer.data
        )[
            idx
        ]


        # Copy the corresponding point-display sizes.
        view_pts_layer.size = np.asarray(
            pts_layer.size
        )[
            idx
        ]


        # Copy every selected point property.
        view_props = {
            k: np.asarray(v)[idx]
            for (
                k,
                v
            ) in p.items()
        }


        # Assign the filtered properties to the view layer.
        view_pts_layer.properties = view_props


        # Recalculate colors for the filtered points.
        _refresh_point_colors(
            view_pts_layer
        )


        # Show or hide the filtered point layer.
        view_pts_layer.visible = bool(
            show_view_layer
        )


    # ========================================================
    # HELPER: REAPPLY THE CURRENT FILTER
    # ========================================================

    def _reapply_current_filter():
        """
        Reapply the active filter after point properties change.
        """

        # Stop when no filter has been initialized.
        if _filter_state[
            "cell_ids"
        ] is None:
            return


        # Update the filtered label layer when enabled.
        if _filter_state[
            "show_filtered_labels"
        ]:

            _update_filtered_labels(
                _filter_state[
                    "cell_ids"
                ]
            )

        else:

            # Clear and hide the filtered label layer.
            _update_filtered_labels(
                []
            )


        # Reapply the point filter using all stored settings.
        _apply_points_filter(
            _filter_state[
                "cell_ids"
            ],
            location_mode=_filter_state[
                "location"
            ],
            lys_ids=_filter_state[
                "lys_ids"
            ],
            show_view_layer=_filter_state[
                "show_view_layer"
            ],
        )

    # ========================================================
    # CREATE EDITABLE LYSOSOME POINT LAYERS
    #
    # Continue only when:
    #   1. Napari table editing is enabled.
    #   2. df is a pandas DataFrame.
    #   3. df contains at least one lysosome row.
    # ========================================================

    if (
        EDIT_LYSOSOME_TABLE_IN_NAPARI
        and isinstance(df, pd.DataFrame)
        and len(df) > 0
    ):

        # Copy the lysosome table and make sure all columns
        # required by the editing system exist.
        df_edit = attach_all_blob_fields(
            df.copy()
        )


        # ====================================================
        # MAP TOTAL CELL SIGNAL TO EACH LYSOSOME
        # ====================================================

        # Check whether the per-cell signal table exists
        # in the current global namespace.
        if "df_signal" in globals():

            # Create a lookup dictionary:
            #
            # original cell ID -> total cell signal.
            cell_signal_dict = dict(
                zip(
                    df_signal["cell_id"],
                    df_signal["cell_signal_total"]
                )
            )

            # Assign the corresponding total cell signal
            # to every lysosome using its original cell ID.
            #
            # Unmatched or outside lysosomes receive 0.0.
            df_edit["cell_signal_total"] = (
                df_edit["cell_id_ch2"]
                .map(cell_signal_dict)
                .fillna(0.0)
            )

        else:

            # Warn that the cell-signal table is unavailable.
            print(
                "[Warning] df_signal missing"
            )

            # Create the signal column with zeros.
            df_edit["cell_signal_total"] = 0.0


        # ====================================================
        # ADD CELL AND SIGNAL FEATURES TO LYSOSOME TABLE
        # ====================================================

        # Check again that df_signal exists.
        if "df_signal" in globals():

            # Create lookup:
            #
            # cell ID -> total cell signal.
            cell_signal_dict = dict(
                zip(
                    df_signal["cell_id"],
                    df_signal["cell_signal_total"]
                )
            )

            # Create lookup:
            #
            # cell ID -> total lysosome-core signal.
            lys_signal_dict = dict(
                zip(
                    df_signal["cell_id"],
                    df_signal["lysosome_core_signal"]
                )
            )

            # Create lookup:
            #
            # cell ID -> quantified cell volume.
            cell_vol_dict = dict(
                zip(
                    df_signal["cell_id"],
                    df_signal["cell_volume_um3"]
                )
            )

            # Create lookup:
            #
            # cell ID -> lysosome-core volume.
            lys_vol_dict = dict(
                zip(
                    df_signal["cell_id"],
                    df_signal["lysosome_core_volume_um3"]
                )
            )


            # Map total cell signal to each lysosome.
            df_edit["cell_signal_total"] = (
                df_edit["cell_id_ch2"]
                .map(cell_signal_dict)
                .fillna(0.0)
            )


            # Map total lysosome-core signal for the
            # assigned cell to each lysosome row.
            df_edit["lysosome_core_signal"] = (
                df_edit["cell_id_ch2"]
                .map(lys_signal_dict)
                .fillna(0.0)
            )


            # Map quantified cell volume to every lysosome.
            df_edit["cell_volume_um3"] = (
                df_edit["cell_id_ch2"]
                .map(cell_vol_dict)
                .fillna(0.0)
            )


            # Map total lysosome-core volume for the cell.
            df_edit["lysosome_core_volume_um3"] = (
                df_edit["cell_id_ch2"]
                .map(lys_vol_dict)
                .fillna(0.0)
            )

        else:

            # Warn that signal features could not be added.
            print(
                "[Warning] df_signal not found — "
                "signal features not added."
            )


        # ====================================================
        # CONVERT LYSOSOME POSITIONS TO NAPARI COORDINATES
        # ====================================================

        # Create an N × 3 point-coordinate array.
        #
        # Napari expects coordinates in:
        #
        # Z, Y, X order.
        pts_zyx = np.stack(
            [
                # Convert physical Z positions to slice units.
                (
                    df_edit["z_um"].to_numpy()
                    / vz_um
                ),

                # Convert physical Y positions to pixel units.
                (
                    df_edit["y_um"].to_numpy()
                    / vy_um
                ),

                # Convert physical X positions to pixel units.
                (
                    df_edit["x_um"].to_numpy()
                    / vx_um
                ),
            ],
            axis=1
        ).astype(np.float32)


        # ====================================================
        # DETERMINE POINT DISPLAY SIZES
        # ====================================================

        # Check whether physical radius information exists.
        if "radius_um" in df_edit.columns:

            # Convert radius from micrometers to equivalent
            # XY voxel units.
            radii_vox = (
                df_edit["radius_um"].to_numpy(
                    dtype=float
                )
                / (
                    np.sqrt(vx_um * vy_um)
                    + 1e-12
                )
            )

            # Napari point size is approximately a diameter,
            # so multiply radius by two.
            #
            # A minimum visible size of two pixels is enforced.
            sizes = np.clip(
                radii_vox * 2,
                2,
                None
            ).astype(np.float32)

        else:

            # If radius information is unavailable,
            # use a constant point size of six pixels.
            sizes = np.full(
                (
                    pts_zyx.shape[0],
                ),
                6,
                dtype=np.float32
            )


        # ====================================================
        # ENSURE CLASSIFICATION AND SERIAL-ID COLUMNS EXIST
        # ====================================================

        # Add the location column if missing.
        if "location_ch2" not in df_edit.columns:

            # Default every lysosome to outside.
            df_edit["location_ch2"] = "outside"


        # Prefer an existing serial cell-ID column.
        if "cell_id_serial" in df_edit.columns:

            # Replace missing values with zero and convert to int.
            df_edit["cell_id_serial"] = (
                df_edit["cell_id_serial"]
                .fillna(0)
                .astype(int)
            )

        # Otherwise use the visualization cell-ID column.
        elif "cell_id_ch2_viz" in df_edit.columns:

            df_edit["cell_id_serial"] = (
                df_edit["cell_id_ch2_viz"]
                .fillna(0)
                .astype(int)
            )

        else:

            # Otherwise map original cell IDs to serial IDs.
            df_edit["cell_id_serial"] = (
                df_edit["cell_id_ch2"]
                .map(cell_id_map_viz)
                .fillna(0)
                .astype(int)
            )


        # Add diameter column if missing.
        if "diameter_um" not in df_edit.columns:
            df_edit["diameter_um"] = np.nan


        # Add peak-intensity column if missing.
        if "peak_gray" not in df_edit.columns:
            df_edit["peak_gray"] = np.nan


        # ====================================================
        # HELPER: RECOMPUTE PER-CELL LYSOSOME SERIAL IDS
        # ====================================================

        def _recompute_lys_id_serial(
            cell_id_serial_arr,
            location_arr
        ):
            """
            Assign sequential lysosome IDs within each serial cell.

            Only points classified as "cell" with a positive
            serial cell ID receive a lysosome serial number.

            Outside or unassigned lysosomes receive zero.
            """

            # Determine the number of lysosome rows.
            n = int(
                len(cell_id_serial_arr)
            )

            # Initialize all lysosome serial IDs to zero.
            out = np.zeros(
                n,
                dtype=int
            )


            # Select intracellular rows with valid serial cell IDs.
            mask = (
                (
                    location_arr.astype(str)
                    == "cell"
                )
                & (
                    cell_id_serial_arr.astype(int)
                    > 0
                )
            )


            # Return all zeros if no valid intracellular
            # lysosomes are present.
            if not np.any(mask):
                return out


            # Build a temporary table used for sorting.
            tmp = pd.DataFrame({

                # Preserve each row's original positional index.
                "idx": np.arange(
                    n,
                    dtype=int
                ),

                # Store serial cell IDs.
                "cell_id_serial":
                    cell_id_serial_arr.astype(int),

                # Store physical Z coordinates.
                "z_um":
                    df_edit["z_um"].to_numpy(
                        dtype=float
                    ),

                # Store physical Y coordinates.
                "y_um":
                    df_edit["y_um"].to_numpy(
                        dtype=float
                    ),

                # Store physical X coordinates.
                "x_um":
                    df_edit["x_um"].to_numpy(
                        dtype=float
                    ),
            })


            # Keep only intracellular rows and sort them
            # consistently within each cell.
            #
            # mergesort is stable, meaning equal values preserve
            # their previous order.
            tmp = (
                tmp.loc[
                    mask
                ]
                .sort_values(
                    [
                        "cell_id_serial",
                        "z_um",
                        "y_um",
                        "x_um"
                    ],
                    kind="mergesort"
                )
            )


            # Assign numbers beginning at one within each cell.
            serial = (
                tmp.groupby(
                    "cell_id_serial"
                )
                .cumcount()
                .to_numpy()
                + 1
            ).astype(int)


            # Write serial IDs back into their original row positions.
            out[
                tmp["idx"].to_numpy(
                    dtype=int
                )
            ] = serial


            # Return the complete serial-ID array.
            return out


        # ====================================================
        # CREATE INITIAL POINT PROPERTIES
        # ====================================================

        # Read initial location classifications.
        loc0 = (
            df_edit["location_ch2"]
            .astype(str)
            .to_numpy()
        )


        # Read initial serial cell IDs.
        cell_serial0 = (
            df_edit["cell_id_serial"]
            .astype(int)
            .to_numpy()
        )


        # Calculate initial per-cell lysosome serial IDs.
        lys_serial0 = _recompute_lys_id_serial(
            cell_serial0,
            loc0
        )


        # Add serial lysosome IDs to the editing DataFrame.
        df_edit["lys_id_serial"] = lys_serial0


        # Build the property dictionary used by Napari.
        props = {

            # Cell/outside classification.
            "location_ch2":
                loc0,

            # Serial cell ID.
            "cell_id_serial":
                cell_serial0,

            # Serial lysosome ID within the cell.
            "lys_id_serial":
                lys_serial0,

            # Physical diameter.
            "diameter_um":
                df_edit["diameter_um"].to_numpy(
                    dtype=float
                ),

            # Local peak intensity.
            "peak_gray":
                df_edit["peak_gray"].to_numpy(
                    dtype=float
                ),

            # Z slice identifier.
            "slice":
                df_edit["slice"].to_numpy(
                    dtype=int
                ),
        }


        # ====================================================
        # EDITABLE LYSOSOME POINTS LAYER
        # ====================================================

        # Add all lysosomes as an editable Napari Points layer.
        pts_layer = viewer.add_points(
            pts_zyx,

            # Set one display size per point.
            size=sizes,

            # Layer name.
            name="Lysosomes (EDIT TABLE)",

            # Attach editable properties.
            properties=props,
        )


        # Draw a black border around every point.
        pts_layer.edge_color = "black"


        # Set the border width.
        pts_layer.edge_width = 0.3


        # Start the point layer in selection mode.
        pts_layer.mode = "select"


        # Display text as:
        #
        # cell serial ID : lysosome serial ID
        pts_layer.text = {
            "string":
                "{cell_id_serial}:{lys_id_serial}",

            "size":
                10,

            "color":
                "white"
        }


        # Assign point face colors based on cell assignment.
        _refresh_point_colors(
            pts_layer
        )


        # ====================================================
        # FILTERED VIEW-ONLY POINTS LAYER
        # ====================================================

        # Create an initially empty Points layer.
        view_pts_layer = viewer.add_points(

            # Use an empty coordinate array with three columns.
            pts_zyx[:0],

            # Use an empty point-size array.
            size=np.array(
                [],
                dtype=np.float32
            ),

            # Layer name.
            name="Lysosomes (VIEW FILTER)",

            # Create empty versions of all properties.
            properties={
                k: np.asarray(v)[:0]
                for (
                    k,
                    v
                ) in props.items()
            },
        )


        # Draw black borders.
        view_pts_layer.edge_color = "black"


        # Use a slightly thinner edge.
        view_pts_layer.edge_width = 0.2


        # Prevent editing; use pan/zoom interaction.
        view_pts_layer.mode = "pan_zoom"


        # Keep the layer hidden until a filter is applied.
        view_pts_layer.visible = False


        # Show the same cell:lysosome text labels.
        view_pts_layer.text = {
            "string":
                "{cell_id_serial}:{lys_id_serial}",

            "size":
                10,

            "color":
                "white"
        }


        # ====================================================
        # HELPER: REFRESH SERIAL IDS AND COLORS
        # ====================================================

        def _refresh_labels_and_serials():
            """
            Recompute per-cell lysosome IDs after edits,
            refresh point colors, and reapply active filters.
            """

            # Stop if the editable points layer is unavailable.
            if pts_layer is None:
                return


            # Copy the current property dictionary.
            p = dict(
                pts_layer.properties
            )


            # Read current location assignments.
            loc = np.array(
                p["location_ch2"]
            ).astype(str)


            # Read current serial cell IDs.
            cell_serial = np.array(
                p["cell_id_serial"]
            ).astype(int)


            # Recalculate lysosome IDs within each cell.
            lys_serial = _recompute_lys_id_serial(
                cell_serial,
                loc
            )


            # Replace the lysosome serial property.
            p["lys_id_serial"] = lys_serial


            # Write all updated properties back to the layer.
            pts_layer.properties = p


            # Refresh point colors.
            _refresh_point_colors(
                pts_layer
            )


            # Reapply any active filters.
            _reapply_current_filter()


        # ====================================================
        # HELPER: UPDATE SELECTED POINT PROPERTIES
        # ====================================================

        def _apply_props_update(
            indices,
            new_loc=None,
            new_cell_serial=None
        ):
            """
            Update location and/or serial cell ID for selected points.
            """

            # Stop when the point layer does not exist
            # or no point indices were supplied.
            if (
                pts_layer is None
                or not indices
            ):
                return


            # Copy the current properties.
            p = dict(
                pts_layer.properties
            )


            # Read location values as editable Python objects.
            loc = np.array(
                p["location_ch2"]
            ).astype(object)


            # Read serial cell IDs as integers.
            cell_serial = np.array(
                p["cell_id_serial"]
            ).astype(int)


            # Process every selected point index.
            for i in indices:

                # Replace location when requested.
                if new_loc is not None:
                    loc[i] = str(
                        new_loc
                    )


                # Replace serial cell ID when requested.
                if new_cell_serial is not None:

                    # Convert the supplied value to int.
                    sid = int(
                        new_cell_serial
                    )

                    # Assign it to the selected point.
                    cell_serial[i] = sid


                # Any point marked outside must have cell ID zero.
                if new_loc == "outside":
                    cell_serial[i] = 0


            # Store updated location values.
            p["location_ch2"] = loc.astype(str)


            # Store updated serial cell IDs.
            p["cell_id_serial"] = (
                cell_serial.astype(int)
            )


            # Write properties back to the point layer.
            pts_layer.properties = p


            # Recalculate lysosome IDs and refresh colors.
            _refresh_labels_and_serials()


        # ====================================================
        # KEYBOARD EDIT: ASSIGN SELECTED POINTS TO CELL
        #
        # Press A while the cursor is over a serial cell label.
        # ====================================================

        @viewer.bind_key("A")
        def assign_selected_to_id_under_cursor(
            event=None
        ):
            """
            Assign selected lysosomes to the serial cell ID
            located under the Napari cursor.
            """

            # Stop if the editable point layer is unavailable.
            if pts_layer is None:
                return


            # Read selected point indices and sort them.
            sel = sorted(
                list(
                    pts_layer.selected_data
                )
            )


            # Require at least one selected point.
            if not sel:

                print(
                    "[Napari edit] No points selected."
                )

                return


            # Read the current cursor position in Z, Y, X order.
            zf, yf, xf = viewer.cursor.position


            # Round cursor coordinates to integer voxels.
            zz = int(
                round(zf)
            )

            yy = int(
                round(yf)
            )

            xx = int(
                round(xf)
            )


            # Check whether the cursor lies inside the label image.
            if not (
                0 <= zz < cell_seg_viz.shape[0]
                and 0 <= yy < cell_seg_viz.shape[1]
                and 0 <= xx < cell_seg_viz.shape[2]
            ):

                print(
                    "[Napari edit] Cursor out of bounds."
                )

                return


            # Read the serial cell ID at the cursor position.
            serial_id = int(
                cell_seg_viz[
                    zz,
                    yy,
                    xx
                ]
            )


            # Require the cursor to lie over a labeled cell.
            if serial_id <= 0:

                print(
                    "[Napari edit] Cursor not over "
                    "a labeled SERIAL ID."
                )

                return


            # Assign all selected lysosomes to this cell.
            _apply_props_update(
                sel,
                new_loc="cell",
                new_cell_serial=serial_id
            )


            # Report the edit.
            print(
                f"[Napari edit] Assigned "
                f"{len(sel)} lysosomes -> "
                f"serial ID {serial_id}"
            )


        # ====================================================
        # KEYBOARD EDIT: MARK SELECTED POINTS OUTSIDE
        #
        # Press X to classify selected lysosomes as outside.
        # ====================================================

        @viewer.bind_key("X")
        def mark_selected_outside(
            event=None
        ):
            """
            Mark selected lysosomes as outside all cells.
            """

            # Stop if the points layer is unavailable.
            if pts_layer is None:
                return


            # Read selected point indices.
            sel = sorted(
                list(
                    pts_layer.selected_data
                )
            )


            # Require at least one selected point.
            if not sel:

                print(
                    "[Napari edit] No points selected."
                )

                return


            # Mark the selected lysosomes outside
            # and assign serial cell ID zero.
            _apply_props_update(
                sel,
                new_loc="outside",
                new_cell_serial=0
            )


            # Report the number of changed points.
            print(
                f"[Napari edit] Marked "
                f"{len(sel)} lysosomes as outside"
            )


        # ====================================================
        # HELPER: EXPORT EDITED CSV
        # ====================================================

        def _export_edited_csv(path):
            """
            Export the current editable point properties
            into a lysosome CSV file.
            """

            # Stop if the editable point layer is unavailable.
            if pts_layer is None:
                return


            # Read current point properties.
            p = pts_layer.properties


            # Start with a copy of the editing DataFrame.
            df_out = df_edit.copy()


            # Replace location classification with edited values.
            df_out["location_ch2"] = np.array(
                p["location_ch2"]
            ).astype(str)


            # Replace serial cell IDs with edited values.
            df_out["cell_id_serial"] = np.array(
                p["cell_id_serial"]
            ).astype(int)


            # Replace serial lysosome IDs with current values.
            df_out["lys_id_serial"] = np.array(
                p["lys_id_serial"]
            ).astype(int)


            # Ensure all required output columns exist
            # and have suitable types.
            df_out = attach_all_blob_fields(
                df_out
            )


            # Save the edited table.
            df_out.to_csv(
                outpath(path),
                index=False
            )


            # Print the output location.
            print(
                f"[Napari edit] Saved edited "
                f"lysosome table: {outpath(path)}"
            )


        # ====================================================
        # KEYBOARD SAVE
        #
        # Press S to save the current edited table.
        # ====================================================

        @viewer.bind_key("S")
        def save_now(
            event=None
        ):

            # Export using the configured edited CSV filename.
            _export_edited_csv(
                LYSOSOME_EDITED_CSV
            )


        # ====================================================
        # MAGICGUI FILTER AND SELECTION PANELS
        # ====================================================

        try:

            # Import the magicgui decorator.
            from magicgui import magicgui


            # ------------------------------------------------
            # FILTER PANEL
            # ------------------------------------------------

            @magicgui(
                # Text shown on the panel's action button.
                call_button="Apply filter",

                # Cell-ID text-entry configuration.
                ids_text={
                    "label":
                        "Cell IDs "
                        "(e.g. all, 1,2,5-8)",

                    "value":
                        "all"
                },

                # Lysosome-ID text-entry configuration.
                lys_text={
                    "label":
                        "Lys IDs in those cells "
                        "(e.g. all, 1,2 or 1-3)",

                    "value":
                        "all"
                },

                # Location-selection dropdown.
                location={
                    "choices":
                        [
                            "both",
                            "cell",
                            "outside"
                        ],

                    "value":
                        "both"
                },

                # Checkbox controlling the filtered label layer.
                show_filtered_labels={
                    "label":
                        "Show filtered ID labels layer",

                    "value":
                        True
                },

                # Checkbox controlling the filtered points layer.
                show_view_layer={
                    "label":
                        "Show VIEW FILTER points layer",

                    "value":
                        True
                },
            )
            def filter_panel(
                ids_text="all",
                lys_text="all",
                location="both",
                show_filtered_labels=True,
                show_view_layer=True
            ):
                """
                Apply cell, location, and lysosome-ID filters.
                """

                # Parse selected serial cell IDs.
                sel_ids = _parse_id_text(
                    ids_text,
                    n_labels
                )


                # Parse selected per-cell lysosome IDs.
                lys_ids = _parse_lys_text(
                    lys_text
                )


                # Store selected cell IDs.
                _filter_state["cell_ids"] = (
                    sel_ids
                )


                # Store selected location mode.
                _filter_state["location"] = str(
                    location
                )


                # Store selected lysosome IDs.
                _filter_state["lys_ids"] = (
                    lys_ids
                )


                # Store filtered-label visibility.
                _filter_state[
                    "show_filtered_labels"
                ] = bool(
                    show_filtered_labels
                )


                # Store filtered-point-layer visibility.
                _filter_state[
                    "show_view_layer"
                ] = bool(
                    show_view_layer
                )


                # Apply all current filter settings.
                _reapply_current_filter()


                # Create readable lysosome-ID text.
                msg_lys = (
                    "all"
                    if lys_ids is None
                    else str(lys_ids)
                )


                # Print the applied filter.
                print(
                    f"[Filter] cells={sel_ids} "
                    f"location={location} "
                    f"lys={msg_lys}"
                )


            # ------------------------------------------------
            # CLEAR FILTER PANEL
            # ------------------------------------------------

            @magicgui(
                call_button="Clear filter"
            )
            def clear_filter_panel():
                """
                Clear all active label and point filters.
                """

                # Mark filter as inactive.
                _filter_state["cell_ids"] = None


                # Reset location filtering.
                _filter_state["location"] = "both"


                # Reset lysosome-ID filtering.
                _filter_state["lys_ids"] = None


                # Reset filtered-label visibility preference.
                _filter_state[
                    "show_filtered_labels"
                ] = True


                # Reset filtered-point visibility preference.
                _filter_state[
                    "show_view_layer"
                ] = True


                # Clear and hide the filtered label layer.
                _update_filtered_labels(
                    []
                )


                # Clear and hide the filtered points layer.
                if view_pts_layer is not None:

                    # Hide the view-only point layer.
                    view_pts_layer.visible = False

                    # Replace its data with an empty point array.
                    view_pts_layer.data = np.zeros(
                        (
                            0,
                            3
                        ),
                        dtype=np.float32
                    )


                # Confirm that the filter was cleared.
                print(
                    "[Filter] Cleared."
                )


            # ------------------------------------------------
            # SELECT POINTS IN EDIT LAYER PANEL
            # ------------------------------------------------

            @magicgui(
                # Button text.
                call_button="Select in EDIT layer",

                # Cell-ID input.
                ids_text={
                    "label":
                        "Cell IDs "
                        "(e.g. 5 or 1,3-4)",

                    "value":
                        "1"
                },

                # Optional lysosome-ID input.
                lys_text={
                    "label":
                        "Lys IDs "
                        "(optional, e.g. all, 1,2)",

                    "value":
                        "all"
                },

                # Location selection.
                location={
                    "choices":
                        [
                            "both",
                            "cell",
                            "outside"
                        ],

                    "value":
                        "cell"
                },
            )
            def select_panel(
                ids_text="1",
                lys_text="all",
                location="cell"
            ):
                """
                Select matching points in the editable layer.
                """

                # Stop when the editable point layer is unavailable.
                if pts_layer is None:
                    return


                # Parse selected cell IDs.
                sel_ids = _parse_id_text(
                    ids_text,
                    n_labels
                )


                # Parse selected lysosome IDs.
                lys_ids = _parse_lys_text(
                    lys_text
                )


                # Copy point properties.
                p = dict(
                    pts_layer.properties
                )


                # Read location classifications.
                loc = np.array(
                    p.get(
                        "location_ch2",
                        []
                    ),
                    dtype=str
                )


                # Read serial cell IDs.
                cid = np.array(
                    p.get(
                        "cell_id_serial",
                        []
                    ),
                    dtype=int
                )


                # Read serial lysosome IDs.
                lys = np.array(
                    p.get(
                        "lys_id_serial",
                        []
                    ),
                    dtype=int
                )


                # Begin by selecting points in the requested cells.
                keep = np.isin(
                    cid,
                    sel_ids
                )


                # Restrict to intracellular points if requested.
                if location == "cell":
                    keep &= (
                        loc == "cell"
                    )


                # Restrict to outside points if requested.
                elif location == "outside":
                    keep &= (
                        loc != "cell"
                    )


                # Apply lysosome-ID filtering when supplied.
                if lys_ids is not None:
                    keep &= np.isin(
                        lys,
                        lys_ids
                    )


                # Convert the Boolean mask into point indices.
                idx = np.where(
                    keep
                )[0]


                # Select the matching points in Napari.
                pts_layer.selected_data = set(
                    int(i)
                    for i in idx.tolist()
                )


                # Make the editable point layer active.
                viewer.layers.selection.active = (
                    pts_layer
                )


                # Create readable lysosome-ID text.
                msg_lys = (
                    "all"
                    if lys_ids is None
                    else str(lys_ids)
                )


                # Report the selection.
                print(
                    f"[Select] Selected {len(idx)} points "
                    f"(cells={sel_ids}, "
                    f"lys={msg_lys}, "
                    f"loc={location})."
                )


            # ------------------------------------------------
            # ADD PANELS TO THE NAPARI WINDOW
            # ------------------------------------------------

            # Add the filter panel to the right side.
            viewer.window.add_dock_widget(
                filter_panel,
                area="right",
                name="Filter cells / lysosomes"
            )


            # Add the selection panel to the right side.
            viewer.window.add_dock_widget(
                select_panel,
                area="right",
                name="Select lysosomes"
            )


            # Add the clear-filter panel to the right side.
            viewer.window.add_dock_widget(
                clear_filter_panel,
                area="right",
                name="Clear filter"
            )


        except Exception as e:

            # Report when magicgui cannot be imported or initialized.
            print(
                "[Napari] magicgui not available; "
                "filter panel disabled. Error:",
                e
            )


    # ========================================================
    # SET INITIAL CAMERA ZOOM
    # ========================================================

    try:

        # Set a modest initial zoom level.
        viewer.camera.zoom = 1.2

    except Exception:

        # Ignore errors from Napari versions or camera modes
        # that do not support this assignment.
        pass


    # ========================================================
    # START THE NAPARI EVENT LOOP
    # ========================================================

    # Open the viewer and keep it running until the user closes it.
    napari.run()


    # ========================================================
    # AUTOMATICALLY SAVE EDITED TABLE AFTER NAPARI CLOSES
    # ========================================================

    # Continue only when editing was enabled and the editable
    # points layer was successfully created.
    if (
        EDIT_LYSOSOME_TABLE_IN_NAPARI
        and pts_layer is not None
    ):

        try:

            # Read the final edited point properties.
            p = pts_layer.properties


            # Start with a fresh copy of the editing DataFrame.
            df_out = df_edit.copy()


            # Store final location classifications.
            df_out["location_ch2"] = np.array(
                p["location_ch2"]
            ).astype(str)


            # Store final serial cell IDs.
            df_out["cell_id_serial"] = np.array(
                p["cell_id_serial"]
            ).astype(int)


            # Store final serial lysosome IDs.
            df_out["lys_id_serial"] = np.array(
                p["lys_id_serial"]
            ).astype(int)


            # Ensure required fields and data types exist.
            df_out = attach_all_blob_fields(
                df_out
            )


            # Save the automatically edited CSV.
            df_out.to_csv(
                outpath(
                    LYSOSOME_EDITED_CSV
                ),
                index=False
            )


            # Report the saved output.
            print(
                f"[Napari edit] Auto-saved edited "
                f"lysosome table: "
                f"{outpath(LYSOSOME_EDITED_CSV)}"
            )

        except Exception as e:

            # Report automatic-save failure.
            print(
                "[Napari edit] Auto-save failed:",
                e
            )


# In[ ]:




