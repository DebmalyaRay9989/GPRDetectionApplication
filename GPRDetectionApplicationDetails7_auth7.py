



"""
 AI-Engine for Buried Object Detection — Streamlit App v8
Model: raw_gpr_objectdetection/1 (Roboflow)
AVNL-OFMK ·  AI-Engine for Buried Object Detection Platform

Improvements over v7:
  • Adaptive upscaling for small GPR scans (e.g. 256×64):
      – Images with longest edge < TARGET_INFER_SIZE (640 px) are upscaled
        before inference so hyperbolic reflections occupy a comparable pixel
        area to what the model was trained on.
  • Multi-scale inference for very small images (longest edge < 320 px):
      – In addition to the primary upscaled pass, also runs inference at ×2
        and ×3 magnification; all detections are merged with cross-class NMS.
        This ensures hyperbolas that are only visible at one zoom level are
        not missed.
  • Aspect-ratio-aware square padding:
      – Short-axis padding with reflected edge content prevents the model's
        receptive field from being starved on very elongated (strip) scans.
  • Adaptive annotation rendering in draw_detections():
      – For tiny images, the annotation canvas is temporarily upscaled to a
        minimum render size so box lines, corner ticks, and label text are
        always legible regardless of input image dimensions.
      – Labels are repositioned below the box when there is no room above.
  • Coordinate remapping verified end-to-end:
      – _remap_preds_to_original() correctly reverses padding + upscale so
        bounding boxes always align with the original input image.
  • New sidebar controls: multi-scale toggle, square-pad toggle.
  • All v7 improvements retained (tiled inference, NMS, download, history, …)
"""

import io
import base64
import json
import os

import csv
import logging
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import requests
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING BACKEND  (ported from Process_sgy_jpeg.py v4)
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.style.use("dark_background")

# ── Signal Processing ─────────────────────────────────────────────────────────

def pp_read_sgy(file_bytes: bytes, filename: str) -> tuple:
    """Read a SEG-Y file from bytes. Returns (data, dt_ns, meta_str)."""
    import tempfile, obspy
    with tempfile.NamedTemporaryFile(suffix=".sgy", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    st_obj = obspy.read(tmp_path)
    os.unlink(tmp_path)
    data  = np.array([tr.data for tr in st_obj], dtype=np.float32).T
    dt_ns = float(st_obj[0].stats.delta) * 1e9
    n_samples, n_traces = data.shape
    meta = (f"File: {filename}  |  Shape: {n_samples}×{n_traces}  |  "
            f"dt: {dt_ns:.4f} ns  |  Fs: {1e3/dt_ns:.1f} MHz  |  "
            f"Range: [{data.min():.4g}, {data.max():.4g}]")
    return data, dt_ns, meta


def pp_dewow(data: np.ndarray, window: int = 10) -> np.ndarray:
    trend = uniform_filter1d(data, size=window, axis=0, mode="reflect")
    return (data - trend).astype(np.float32)


def pp_bandpass(data: np.ndarray, low_MHz: float, high_MHz: float,
                dt_ns: float, order: int = 4) -> np.ndarray:
    fs_MHz = 1e3 / dt_ns
    nyq    = fs_MHz / 2.0
    lo = float(np.clip(low_MHz  / nyq, 1e-4, 0.9999))
    hi = float(np.clip(high_MHz / nyq, 1e-4, 0.9999))
    if lo >= hi:
        return data
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, data, axis=0).astype(np.float32)


def pp_background_removal(data: np.ndarray, mode: str = "mean") -> np.ndarray:
    bg = np.median(data, axis=1, keepdims=True) if mode == "median" \
         else np.mean(data, axis=1, keepdims=True)
    return (data - bg).astype(np.float32)


def pp_trace_normalise(data: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    rms = np.sqrt(np.mean(data ** 2, axis=0, keepdims=True)) + eps
    return (data / rms).astype(np.float32)


def pp_apply_gain(data: np.ndarray, mode: str = "quadratic",
                  gain_db: float = 30.0, agc_window: int = 20
                  ) -> tuple:
    n  = data.shape[0]
    d  = np.linspace(0, 1, n, dtype=np.float32)
    mg = 10 ** (gain_db / 20.0)
    if mode == "linear":
        gv = (1 + (mg - 1) * d).reshape(-1, 1)
    elif mode == "quadratic":
        gv = (1 + (mg - 1) * d ** 2).reshape(-1, 1)
    elif mode == "agc":
        pad       = np.pad(data, ((agc_window, agc_window), (0, 0)), mode="edge")
        local_rms = np.array([
            np.sqrt(np.mean(pad[i:i + 2 * agc_window + 1] ** 2, axis=0))
            for i in range(n)
        ], dtype=np.float32) + 1e-9
        return (data / local_rms).astype(np.float32), np.ones(n, dtype=np.float32)
    else:
        raise ValueError(f"Unknown gain mode: {mode!r}")
    return (data * gv).astype(np.float32), gv.ravel()


# ── Image Helpers ─────────────────────────────────────────────────────────────

def pp_normalise_uint8(data: np.ndarray, plo: float = 1.0, phi: float = 99.0) -> np.ndarray:
    img  = np.nan_to_num(data.copy())
    lo   = float(np.percentile(img, plo))
    hi   = float(np.percentile(img, phi))
    span = hi - lo
    if span < 1e-12:
        return np.zeros_like(img, dtype=np.uint8)
    return np.clip((img - lo) / span * 255, 0, 255).astype(np.uint8)


def pp_apply_clahe(img_u8: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        return clahe.apply(img_u8)
    except ImportError:
        pass
    H, W  = img_u8.shape
    out   = img_u8.copy().astype(np.float32)
    th, tw = max(1, H // tile), max(1, W // tile)
    for r in range(0, H, th):
        for c in range(0, W, tw):
            patch = img_u8[r:r+th, c:c+tw].astype(np.float32)
            lo, hi = np.percentile(patch, 1), np.percentile(patch, 99)
            span = hi - lo
            if span > 1e-3:
                patch = np.clip((patch - lo) / span * 255, 0, 255)
            out[r:r+th, c:c+tw] = patch
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Matplotlib figure → PIL ───────────────────────────────────────────────────

def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()


def pp_plot_bscan(data: np.ndarray, title: str, cmap: str = "gray") -> Image.Image:
    lim = float(np.percentile(np.abs(data), 99))
    fig, ax = plt.subplots(figsize=(11, 3.5))
    im = ax.imshow(data, cmap=cmap, aspect="auto", interpolation="nearest",
                   vmin=-lim, vmax=lim)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Trace #", fontsize=9)
    ax.set_ylabel("Sample #", fontsize=9)
    fig.colorbar(im, ax=ax, label="Amplitude", shrink=0.85, pad=0.02)
    fig.tight_layout()
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def pp_plot_gain_curve(gv: np.ndarray) -> Image.Image:
    samples = np.arange(len(gv))
    gv_db   = 20 * np.log10(np.maximum(gv, 1e-9))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, y, xlabel, col, ttl in [
        (a0, gv,    "Gain (linear)", "deepskyblue", "Linear Scale"),
        (a1, gv_db, "Gain (dB)",     "tomato",      "dB Scale"),
    ]:
        ax.plot(y, samples, color=col, linewidth=1.5)
        ax.invert_yaxis()
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Sample #", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Applied Gain Curve", fontsize=11, fontweight="bold")
    fig.tight_layout()
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def pp_plot_summary(raw: np.ndarray, bg: np.ndarray,
                    gained: np.ndarray, cmap: str = "gray") -> Image.Image:
    fig = plt.figure(figsize=(17, 3.5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.28)
    stages = [(raw, "① Original B-scan"),
              (bg,  "② Background Removed"),
              (gained, "③ Gained")]
    for i, (arr, ttl) in enumerate(stages):
        ax = fig.add_subplot(gs[0, i])
        lim = float(np.percentile(np.abs(arr), 99))
        im  = ax.imshow(arr, cmap=cmap, aspect="auto",
                        interpolation="nearest", vmin=-lim, vmax=lim)
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        ax.set_xlabel("Trace #", fontsize=8)
        ax.set_ylabel("Sample #", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.80, pad=0.02, label="Amplitude")
    fig.suptitle("GPR Processing Pipeline", fontsize=11, fontweight="bold")
    fig.tight_layout()
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


# ── Full single-file pipeline (returns dict of arrays + PIL outputs) ──────────

def pp_run_pipeline(file_bytes: bytes, filename: str, cfg: dict) -> dict:
    """
    Run the full preprocessing pipeline on one SGY file.
    Returns a result dict with intermediate arrays, figures, and the final JPEG bytes.
    """
    result: dict = {"status": "OK", "log": [], "filename": filename}

    def _log(msg: str):
        result["log"].append(msg)

    try:
        # 1. Read
        _log(f"📖 Reading {filename}…")
        raw, dt_ns, meta = pp_read_sgy(file_bytes, filename)
        result["raw"]    = raw
        result["dt_ns"]  = dt_ns
        result["meta"]   = meta
        result["n_samples"], result["n_traces"] = raw.shape
        _log(f"   {meta}")

        # 2. Dewow
        data = raw.copy()
        if cfg["apply_dewow"]:
            _log(f"🔧 Dewow  (window={cfg['dewow_window']})…")
            data = pp_dewow(data, window=cfg["dewow_window"])
        result["after_dewow"] = data.copy()

        # 3. Bandpass
        if cfg["apply_bandpass"]:
            _log(f"🔧 Bandpass  {cfg['bp_low_MHz']:.0f}–{cfg['bp_high_MHz']:.0f} MHz…")
            data = pp_bandpass(data, cfg["bp_low_MHz"], cfg["bp_high_MHz"],
                               dt_ns, cfg["bp_order"])
        result["after_bandpass"] = data.copy()

        # 4. Background removal
        _log(f"🔧 Background removal  (mode={cfg['bg_mode']})…")
        bg = pp_background_removal(data, mode=cfg["bg_mode"])
        result["after_bg"] = bg.copy()

        # 5. Trace normalise
        if cfg["trace_normalise"]:
            _log("🔧 Trace normalisation…")
            bg = pp_trace_normalise(bg)

        # 6. Gain
        _log(f"🔧 Gain  mode={cfg['gain_mode']}  db={cfg['gain_db']:.1f}…")
        gained, gv = pp_apply_gain(bg, mode=cfg["gain_mode"],
                                   gain_db=cfg["gain_db"],
                                   agc_window=cfg["agc_window"])
        result["gained"] = gained
        result["gv"]     = gv

        # 7. Build output JPEG (640×640)
        _log("🖼 Building 640×640 JPEG…")
        img_u8 = pp_normalise_uint8(gained, 1.0, 99.0)
        pil_out = Image.fromarray(img_u8, mode="L")
        w, h    = cfg["resize_shape"]
        pil_sq  = pil_out.resize((w, h), Image.Resampling.LANCZOS)
        buf     = io.BytesIO()
        pil_sq.save(buf, format="JPEG", quality=cfg["jpeg_quality"])
        result["output_jpeg_bytes"] = buf.getvalue()
        result["output_pil"]        = pil_sq

        _log("✅ Pipeline complete.")

    except Exception as exc:
        result["status"] = "FAILED"
        result["log"].append(f"❌ ERROR: {exc}")
        result["log"].append(traceback.format_exc())

    return result


# Default preprocessing config (mirrors Process_sgy_jpeg.py CONFIG)
_PP_DEFAULT_CFG: dict = {
    "gain_mode":        "quadratic",
    "gain_db":          30.0,
    "agc_window":       20,
    "bg_mode":          "mean",
    "apply_dewow":      True,
    "dewow_window":     39,
    "apply_bandpass":   True,
    "bp_low_MHz":       100.0,
    "bp_high_MHz":      900.0,
    "bp_order":         4,
    "trace_normalise":  False,

    "resize_shape":     (640, 640),
    "jpeg_quality":     95,
    "cmap":             "gray",
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Engine for Buried Object Detection · AVNL-OFMK",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION  — Simple username / hashed-password gate
# Passwords are bcrypt-hashed; never stored in plain text.
# To add / remove users, edit the 'credentials' block below and regenerate
# hashes with:  python -c "import bcrypt; print(bcrypt.hashpw(b'pw', bcrypt.gensalt()).decode())"
# ─────────────────────────────────────────────────────────────────────────────
_AUTH_CONFIG: dict = {
    "credentials": {
        "usernames": {
            "GPRAdmin": {
                "name": "GPR Administrator",
                # Password: SSP242312
                "password": "$2b$12$uQWfUK5KDH2SgnJSe8qANekJdMEUVkAR0abNsCsGhx1ODBEbtZsFC",
                "role": "admin",
            },
            "GPRUser": {
                "name": "GPR Operator",
                # Password: GPRUser
                "password": "$2b$12$euUbzy/jpUA4JnG6vFfI/O6CH3gD2clqU24y5ygdFKm4zQKuFOs02",
                "role": "user",
            },
            "RoboGPR": {
                "name": "Robo GPR Agent",
                # Password: RoboGPR
                "password": "$2b$12$xKfYfjNcqbkO2.c9HfJVcua9JmlR7/78ZetlG4.fXCxBji4b8uRpy",
                "role": "user",
            },
        }
    },
    "cookie": {
        "expiry_days": 1,
        "key": "avnl_ofmk_gpr_auth_v8",   # secret signing key — change in production
        "name": "avnl_gpr_session",
    },
}

# Instantiate authenticator (v0.4.x API)
_authenticator = stauth.Authenticate(
    _AUTH_CONFIG["credentials"],
    _AUTH_CONFIG["cookie"]["name"],
    _AUTH_CONFIG["cookie"]["key"],
    _AUTH_CONFIG["cookie"]["expiry_days"],
)

# ── Login screen styling ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* Login page backdrop */
[data-testid="stVerticalBlock"] > div:first-child {
    background: transparent;
}
.login-header {
    text-align: center;
    padding: 36px 0 8px;
    font-family: 'IBM Plex Mono', monospace;
}
.login-logo {
    font-size: 3rem;
    color: #e8e8e8;
    text-shadow: 0 0 30px rgba(220,220,220,0.4);
    letter-spacing: 6px;
}
.login-sub {
    font-size: .72rem;
    color: rgba(200,200,200,0.45);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Render login widget ───────────────────────────────────────────────────────
_login_result = _authenticator.login(location="main")

# Unpack result — v0.4.x returns (name, authentication_status, username)
if isinstance(_login_result, tuple):
    _auth_name, _auth_status, _auth_username = _login_result
else:
    # Some builds expose state via st.session_state directly
    _auth_name     = st.session_state.get("name")
    _auth_status   = st.session_state.get("authentication_status")
    _auth_username = st.session_state.get("username")

if _auth_status is False:
    st.markdown("""
    <div style='text-align:center; font-family:IBM Plex Mono,monospace;
                color:#e05555; font-size:.82rem; letter-spacing:2px;
                margin-top:12px;'>
        ⛔ ACCESS DENIED — Invalid credentials
    </div>""", unsafe_allow_html=True)
    st.stop()

if _auth_status is None:
    # Show branding above the login form
    st.markdown("""
    <div class='login-header'>
        <div class='login-logo'>AVNL-OFMK</div>
        <div class='login-sub'>AI-Engine · Buried Object Detection · Secure Access</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Authenticated — determine role ───────────────────────────────────────────
_current_role = (
    _AUTH_CONFIG["credentials"]["usernames"]
    .get(_auth_username, {})
    .get("role", "user")
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# Load API key from Streamlit secrets or environment variable.
# To run locally without a secrets.toml, set:  export ROBOFLOW_API_KEY=your_key
_FALLBACK_KEY = "reykCRkfdScF0S9rTdJq"

def _load_api_key() -> str:
    try:
        key = st.secrets.get("ROBOFLOW_API_KEY")
        if key:
            return key
    except (FileNotFoundError, KeyError):
        pass
    return os.environ.get("ROBOFLOW_API_KEY") or _FALLBACK_KEY

ROBOFLOW_API_KEY: str = _load_api_key()

MODEL_ID      = "raw_gpr_objectdetection/1"
ROBOFLOW_URL  = f"https://detect.roboflow.com/{MODEL_ID}"
API_TIMEOUT   = 30            # seconds
API_RETRIES   = 3             # number of retry attempts on transient failure
JPEG_QUALITY  = 92

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_REG  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# ── Tiled inference settings ─────────────────────────────────────────────────
# Images are NEVER downscaled below their natural resolution in tiled mode.
# For images whose longest edge exceeds TILE_THRESHOLD, the image is split into
# overlapping tiles (TILE_SIZE × TILE_SIZE) with TILE_OVERLAP px border overlap.
# Detections from every tile are remapped to original-image coordinates and
# duplicate boxes are removed with NMS (IOU_THRESHOLD).
TILE_THRESHOLD = 1280   # px  — run tiled inference above this long-edge size
TILE_SIZE      = 640    # px  — side length of each tile sent to the API
TILE_OVERLAP   = 128    # px  — overlap between adjacent tiles (catches border hyperbolas)
IOU_THRESHOLD  = 0.45   # NMS IOU threshold for merging cross-tile duplicates

# ── Small-image adaptive upscaling ───────────────────────────────────────────
# GPR B-scans with unusual aspect ratios (e.g. 256×64) are extremely small
# relative to the model's native 640 px training resolution.  We upscale such
# images before inference so hyperbolas occupy a comparable number of pixels
# to those seen during training, then map detections back to original coordinates.
#
# Strategy:
#   1. Compute an integer upscale factor so the LONGEST edge reaches TARGET_INFER_SIZE.
#   2. If BOTH edges are below MIN_EDGE_FOR_MULTISCALE we run multi-scale inference
#      (scales ×2 and ×3 in addition to ×1 of the upscaled image) and merge via NMS.
#   3. Pad short images to a square with reflect-padding before sending so the
#      model's receptive field is not starved on the narrow axis.
TARGET_INFER_SIZE      = 640   # px  — target longest-edge after upscaling
MIN_EDGE_FOR_MULTISCALE = 320  # px  — if longest edge < this, also run ×2/×3 scales
PAD_TO_SQUARE           = True # pad narrow images to square before API call

# Each detected class gets a unique colour for its bounding box and card.
# No threat-level classification — every detection is shown equally.
CLASS_META: Dict[str, dict] = {
    "landmine": {"color": "#ff3030", "icon": "💣"},
    "mine":     {"color": "#ff5555", "icon": "💣"},
    "ied":      {"color": "#ff2020", "icon": "💣"},
    "threat":   {"color": "#ff6600", "icon": "⚠️"},
    "metal":    {"color": "#ff8c00", "icon": "🔩"},
    "pipe":     {"color": "#ffa500", "icon": "🔧"},
    "cable":    {"color": "#ffd700", "icon": "⚡"},
    "utility":  {"color": "#ffe066", "icon": "🔌"},
    "rock":     {"color": "#00ffb4", "icon": "🪨"},
    "root":     {"color": "#7ac9a9", "icon": "🌿"},
    "void":     {"color": "#00bfff", "icon": "⭕"},
    "clutter":  {"color": "#aaaaaa", "icon": "📦"},
}
DEFAULT_META: dict = {"color": "#00ff8c", "icon": "❓"}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp {
    background-color: #0d0d0d;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(220,220,220,.03) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(180,180,200,.025) 0%, transparent 50%),
        linear-gradient(rgba(200,200,200,.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(200,200,200,.018) 1px, transparent 1px);
    background-size: auto, auto, 48px 48px, 48px 48px;
}
.stApp::after {
    content:""; position:fixed; inset:0; pointer-events:none; z-index:1000;
    background: repeating-linear-gradient(0deg,transparent,transparent 3px,
                rgba(0,0,0,.04) 3px,rgba(0,0,0,.04) 4px);
}

/* ── Header ── */
.gpr-header {
    display:flex; align-items:center; gap:20px;
    padding:22px 0 16px;
    border-bottom:1px solid rgba(220,220,220,.15);
    margin-bottom:24px;
}
.gpr-logotype {
    font-family:'IBM Plex Mono',monospace; font-size:2.6rem; color:#e8e8e8;
    text-shadow:0 0 20px rgba(220,220,220,0.35),0 0 50px rgba(200,200,200,0.12);
    letter-spacing:4px; line-height:1;
}
.gpr-head-title { font-size:2.6rem; font-weight:700; color:#f0f0f0; margin:0; letter-spacing:.4px; }
.gpr-head-sub   {
    font-family:'IBM Plex Mono',monospace; font-size:.68rem;
    color:rgba(200,200,200,0.45); letter-spacing:4px; text-transform:uppercase;
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    background:rgba(255,255,255,.04) !important;
    border:1px solid rgba(220,220,220,.22) !important;
    color:#e8e8e8 !important; border-radius:7px !important;
    font-family:'IBM Plex Mono',monospace !important; font-size:.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color:#d0d0d0 !important;
    box-shadow:0 0 0 2px rgba(200,200,200,.12) !important;
}
label[data-testid="stWidgetLabel"] p {
    color:#999999 !important; font-size:.72rem !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
    font-family:'IBM Plex Mono',monospace !important;
}

/* ── Buttons ── */
.stButton > button {
    background:linear-gradient(135deg,#d8d8d8,#a8a8a8) !important;
    color:#0d0d0d !important; font-weight:700 !important;
    font-family:'IBM Plex Sans',sans-serif !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
    border:none !important; border-radius:7px !important;
    padding:10px 26px !important; font-size:.82rem !important;
    transition:all .2s !important;
}
.stButton > button:hover {
    box-shadow:0 0 24px rgba(200,200,200,.35) !important;
    transform:translateY(-2px) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background:rgba(10,10,10,.97) !important;
    border-right:1px solid rgba(220,220,220,.1) !important;
}
[data-testid="stSidebar"] * { color:#b8b8b8; }
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button * { color:#000000 !important; }

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#c8c8c8 !important; }

/* ── File uploader ── */
.stFileUploader > div {
    background:rgba(255,255,255,.025) !important;
    border:1px dashed rgba(220,220,220,.22) !important;
    border-radius:10px !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background:rgba(255,255,255,.03);
    border:1px solid rgba(220,220,220,.1);
    border-radius:10px; padding:14px 18px;
}
[data-testid="metric-container"] label {
    color:#888888 !important; font-size:.68rem !important;
    letter-spacing:2px !important; font-family:'IBM Plex Mono',monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#e0e0e0 !important; font-family:'IBM Plex Mono',monospace !important;
    font-size:1.7rem !important;
}

/* ── Cards ── */
.card {
    background:rgba(20,20,20,.85);
    border:1px solid rgba(220,220,220,.1);
    border-radius:10px; padding:20px 24px; margin-bottom:12px;
    backdrop-filter:blur(8px);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(255,255,255,.03) !important;
    border-radius:8px; gap:2px; padding:4px;
}
.stTabs [data-baseweb="tab"] {
    font-family:'IBM Plex Mono',monospace !important; font-size:.75rem !important;
    letter-spacing:2px !important; color:#888888 !important;
    border-radius:6px !important; padding:8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background:rgba(220,220,220,.1) !important; color:#e0e0e0 !important;
}

/* ── Alerts ── */
.stAlert {
    background:rgba(255,255,255,.04) !important;
    border:1px solid rgba(220,220,220,.15) !important;
    border-radius:8px !important;
}

/* ── Progress ── */
.stProgress > div > div { background:#c0c0c0 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    color:#888888 !important; font-family:'IBM Plex Mono',monospace !important;
    font-size:.75rem !important; letter-spacing:1px !important;
}

/* ── Helpers ── */
.sec-label {
    font-family:'IBM Plex Mono',monospace; font-size:.68rem;
    color:rgba(200,200,200,0.4); letter-spacing:4px; text-transform:uppercase; margin-bottom:10px;
}
.mono { font-family:'IBM Plex Mono',monospace; }

@keyframes pulse {
    0%,100%{box-shadow:0 0 0 0 rgba(200,200,200,.4)}
    50%{box-shadow:0 0 0 7px rgba(200,200,200,0)}
}
.dot-live {
    display:inline-block; width:8px; height:8px; background:#c8c8c8;
    border-radius:50%; margin-right:7px;
    animation:pulse 1.8s ease-in-out infinite; vertical-align:middle;
}

.empty-state {
    background:rgba(255,255,255,.015); border:1px dashed rgba(220,220,220,.12);
    border-radius:10px; padding:52px 20px; text-align:center;
}

hr { border-color:rgba(220,220,220,.1) !important; }

/* ── Image parity: prevent Streamlit from auto-stretching images ── */
[data-testid="stImage"] img {
    max-width: 100% !important;
    height: auto !important;
    display: block !important;
}
[data-testid="column"] [data-testid="stImage"] {
    width: 100% !important;
}
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0d0d0d; }
::-webkit-scrollbar-thumb { background:rgba(200,200,200,.2); border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in [("last_preds", []), ("last_image", None),
               ("scan_history", []), ("total_scans", 0)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_meta(cls_name: str) -> dict:
    """Return display metadata for a detection class name."""
    low = cls_name.lower()
    for k, v in CLASS_META.items():
        if k in low:
            return v
    return DEFAULT_META


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _to_rgb(image: Image.Image) -> Image.Image:
    """Flatten alpha / palette and return a clean RGB image (no resize)."""
    if image.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", image.size, (0, 0, 0))
        bg.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[3])
        return bg
    return image.convert("RGB")


def _encode_jpeg(image: Image.Image) -> str:
    """Return base-64 JPEG string for the given PIL image."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL API CALL  (single image, with retry)
# ─────────────────────────────────────────────────────────────────────────────
def _call_api(img_b64: str, confidence: int, overlap: int) -> Dict[str, Any]:
    """POST one base-64 image to Roboflow and return the parsed JSON."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, API_RETRIES + 1):
        try:
            resp = requests.post(
                ROBOFLOW_URL,
                params={
                    "api_key":    ROBOFLOW_API_KEY,
                    "confidence": confidence,
                    "overlap":    overlap,
                },
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=API_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < API_RETRIES:
                time.sleep(2 ** (attempt - 1))
        except requests.exceptions.HTTPError:
            raise
    raise RuntimeError(f"API unreachable after {API_RETRIES} attempts: {last_exc}")


# ─────────────────────────────────────────────────────────────────────────────
# NMS HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a: dict, b: dict) -> float:
    """Intersection-over-Union for two centre-format prediction dicts."""
    ax1, ay1 = a["x"] - a["width"] / 2,  a["y"] - a["height"] / 2
    ax2, ay2 = a["x"] + a["width"] / 2,  a["y"] + a["height"] / 2
    bx1, by1 = b["x"] - b["width"] / 2,  b["y"] - b["height"] / 2
    bx2, by2 = b["x"] + b["width"] / 2,  b["y"] + b["height"] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _nms(predictions: List[dict], iou_thresh: float = IOU_THRESHOLD) -> List[dict]:
    """
    Greedy NMS over a list of Roboflow-format prediction dicts.
    Keeps the highest-confidence box when two boxes of the same class overlap.
    """
    if not predictions:
        return []
    by_class: Dict[str, List[dict]] = {}
    for p in predictions:
        by_class.setdefault(p.get("class", ""), []).append(p)
    kept: List[dict] = []
    for cls_preds in by_class.values():
        cls_preds = sorted(cls_preds, key=lambda p: p.get("confidence", 0), reverse=True)
        while cls_preds:
            best = cls_preds.pop(0)
            kept.append(best)
            cls_preds = [p for p in cls_preds if _iou(best, p) < iou_thresh]
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# TILED INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def _infer_tile(tile: Image.Image, offset_x: int, offset_y: int,
                confidence: int, overlap: int) -> List[dict]:
    """
    Run inference on one tile and remap box coordinates to full-image space.
    Boxes are clamped to the tile boundaries before the offset is applied so
    detections from a tile never land outside the full image.
    """
    tw, th = tile.size
    result = _call_api(_encode_jpeg(tile), confidence, overlap)
    preds  = result.get("predictions", [])
    out: List[dict] = []
    for p in preds:
        cx = max(p["width"]  / 2, min(tw - p["width"]  / 2, p["x"]))
        cy = max(p["height"] / 2, min(th - p["height"] / 2, p["y"]))
        new_p       = dict(p)
        new_p["x"]  = cx + offset_x
        new_p["y"]  = cy + offset_y
        out.append(new_p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SMALL-IMAGE ADAPTIVE UPSCALING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _upscale_for_inference(rgb: Image.Image,
                            target: int = TARGET_INFER_SIZE,
                            pad_square: bool = PAD_TO_SQUARE
                            ) -> Tuple[Image.Image, float, int, int]:
    """
    Upscale a small image so its longest edge reaches *target* pixels.

    Optionally pads the shorter axis with reflected content so the model sees
    a square-ish canvas rather than a very elongated strip (critical for images
    like 256×64 where the model's receptive field is starved on the short axis).

    Returns:
        prepared_img  — the upscaled (and optionally padded) PIL image
        scale         — the scale factor applied (original × scale = upscaled)
        pad_left      — pixels of left/right padding added (for coord remapping)
        pad_top       — pixels of top/bottom padding added (for coord remapping)
    """
    W, H = rgb.size
    longest = max(W, H)

    # Compute integer scale so longest edge ≥ target
    scale = max(1.0, target / longest)

    new_w = max(1, round(W * scale))
    new_h = max(1, round(H * scale))
    upscaled = rgb.resize((new_w, new_h), Image.LANCZOS)

    pad_left = pad_top = 0

    if pad_square:
        side = max(new_w, new_h)
        pad_left = (side - new_w) // 2
        pad_top  = (side - new_h) // 2
        # Reflect-pad: fill borders with mirrored edge content (better than black
        # for GPR where edge artefacts could create false hyperbolas on black bg)
        padded = Image.new("RGB", (side, side), (0, 0, 0))
        padded.paste(upscaled, (pad_left, pad_top))

        # Fill padding regions with reflected pixels rather than black
        if pad_left > 0:
            left_strip  = upscaled.crop((0, 0, pad_left, new_h)).transpose(Image.FLIP_LEFT_RIGHT)
            right_strip = upscaled.crop((new_w - (side - new_w - pad_left), 0,
                                          new_w, new_h)).transpose(Image.FLIP_LEFT_RIGHT)
            padded.paste(left_strip,  (0, pad_top))
            padded.paste(right_strip, (pad_left + new_w, pad_top))
        if pad_top > 0:
            top_strip    = upscaled.crop((0, 0, new_w, pad_top)).transpose(Image.FLIP_TOP_BOTTOM)
            bottom_strip = upscaled.crop((0, new_h - (side - new_h - pad_top),
                                           new_w, new_h)).transpose(Image.FLIP_TOP_BOTTOM)
            padded.paste(top_strip,    (pad_left, 0))
            padded.paste(bottom_strip, (pad_left, pad_top + new_h))
        return padded, scale, pad_left, pad_top

    return upscaled, scale, 0, 0


def _remap_preds_to_original(preds: List[dict],
                              scale: float,
                              pad_left: int,
                              pad_top: int,
                              orig_w: int,
                              orig_h: int) -> List[dict]:
    """
    Convert box coordinates from the upscaled/padded inference space back to the
    original image pixel space, clamping boxes to image bounds.
    """
    remapped = []
    for p in preds:
        # Remove padding offset first, then reverse the upscale
        cx = (p["x"] - pad_left) / scale
        cy = (p["y"] - pad_top)  / scale
        bw = p["width"]          / scale
        bh = p["height"]         / scale

        # Clamp centre so box stays inside the original image
        cx = max(bw / 2, min(orig_w - bw / 2, cx))
        cy = max(bh / 2, min(orig_h - bh / 2, cy))

        new_p = dict(p)
        new_p["x"]      = cx
        new_p["y"]      = cy
        new_p["width"]  = min(bw, orig_w)
        new_p["height"] = min(bh, orig_h)
        remapped.append(new_p)
    return remapped


def _infer_at_scale(rgb: Image.Image,
                    scale_factor: float,
                    confidence: int,
                    overlap: int,
                    pad_square: bool = PAD_TO_SQUARE) -> List[dict]:
    """
    Upscale *rgb* by *scale_factor*, run inference, and remap detections back
    to the original *rgb* pixel space.  Used for multi-scale inference on very
    small images.
    """
    orig_w, orig_h = rgb.size
    if scale_factor <= 1.0:
        prepared, scale, pl, pt = _upscale_for_inference(rgb, TARGET_INFER_SIZE, pad_square)
    else:
        # Explicit fractional scale (e.g. 2× or 3×) independent of target size
        new_w = round(orig_w * scale_factor)
        new_h = round(orig_h * scale_factor)
        upscaled = rgb.resize((new_w, new_h), Image.LANCZOS)
        prepared, extra_scale, pl, pt = _upscale_for_inference(
            upscaled, max(new_w, new_h, TARGET_INFER_SIZE), pad_square)
        # Combine the two scale factors
        scale = scale_factor * extra_scale

    result = _call_api(_encode_jpeg(prepared), confidence, overlap)
    preds  = result.get("predictions", [])
    return _remap_preds_to_original(preds, scale, pl, pt, orig_w, orig_h)


def run_inference(image: Image.Image, confidence: int, overlap: int,
                  tile: bool = True, tile_px: int = TILE_SIZE,
                  tile_ov: int = TILE_OVERLAP,
                  multi_scale: bool = True,
                  pad_square: bool = PAD_TO_SQUARE) -> Dict[str, Any]:
    """
    Adaptive full-resolution inference pipeline for GPR B-scans of ANY size.

    SIZE ROUTING LOGIC
    ──────────────────
    Case A — SMALL image (longest edge < TARGET_INFER_SIZE):
        The image is upscaled so its longest edge reaches TARGET_INFER_SIZE before
        being sent to the model (YOLO was trained on ~640 px images; a raw 256×64
        crop produces hyperbolas only 5–10 px tall that the model cannot recognise).
        If the longest edge is also < MIN_EDGE_FOR_MULTISCALE, multi-scale inference
        is additionally run at ×2 and ×3 magnification; all results are merged with NMS
        to catch hyperbolas that are only visible at one specific zoom level.
        Detected boxes are mapped back to original-image coordinates before returning.

    Case B — MEDIUM image (TARGET_INFER_SIZE ≤ longest edge ≤ TILE_THRESHOLD):
        Sent as-is with a single API call.  No upscaling, no tiling.

    Case C — LARGE image (longest edge > TILE_THRESHOLD):
        Split into TILE_SIZE × TILE_SIZE overlapping tiles (stride = TILE_SIZE − TILE_OVERLAP).
        Each tile is sent at native resolution.  Boxes from all tiles are remapped to
        full-image coordinates and cross-border duplicates are removed with NMS.

    Returns a dict with key "predictions" whose boxes are in the ORIGINAL image's
    coordinate space — ready for draw_detections() without further adjustment.
    """
    rgb  = _to_rgb(image)
    W, H = rgb.size
    longest = max(W, H)

    # ── Case A: Small image — upscale first ──────────────────────────────────
    if longest < TARGET_INFER_SIZE:
        all_preds: List[dict] = []

        # Primary pass: upscale to TARGET_INFER_SIZE with optional square padding
        all_preds.extend(_infer_at_scale(rgb, 1.0, confidence, overlap, pad_square))

        # Multi-scale passes for very small images
        if multi_scale and longest < MIN_EDGE_FOR_MULTISCALE:
            for extra in (2.0, 3.0):
                try:
                    all_preds.extend(_infer_at_scale(rgb, extra, confidence, overlap, pad_square))
                except Exception:
                    pass   # one failing scale should not abort the whole inference

        merged = _nms(all_preds)
        return {"predictions": merged, "image": {"width": W, "height": H}}

    # ── Case B: Medium image — single API call ────────────────────────────────
    if not tile or longest <= TILE_THRESHOLD:
        result = _call_api(_encode_jpeg(rgb), confidence, overlap)
        return result

    # ── Case C: Large image — tiled path ─────────────────────────────────────
    stride = tile_px - tile_ov
    all_preds = []
    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))

    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_px, W)
            y1 = min(y0 + tile_px, H)
            tx0 = max(0, x1 - tile_px)
            ty0 = max(0, y1 - tile_px)
            tile_img   = rgb.crop((tx0, ty0, tx0 + tile_px, ty0 + tile_px))
            tile_preds = _infer_tile(tile_img, tx0, ty0, confidence, overlap)
            all_preds.extend(tile_preds)

    merged = _nms(all_preds)
    return {"predictions": merged, "image": {"width": W, "height": H}}


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def draw_detections(image: Image.Image, predictions: List[dict],
                    min_render_size: int = 512) -> Image.Image:
    """
    Return a copy of *image* annotated with bounding boxes and labels.

    For very small images the annotation canvas is temporarily upscaled to
    *min_render_size* on its longest edge so that labels and corner ticks are
    legible, then downscaled back to the original dimensions before returning.
    This ensures annotations are always visible regardless of input image size.
    """
    img_rgb = image.convert("RGB")
    orig_w, orig_h = img_rgb.size
    longest = max(orig_w, orig_h)

    # --- Render at a large-enough canvas if input is tiny ─────────────────────
    if longest < min_render_size:
        render_scale = min_render_size / longest
        render_w = max(1, round(orig_w * render_scale))
        render_h = max(1, round(orig_h * render_scale))
        canvas = img_rgb.resize((render_w, render_h), Image.NEAREST)
    else:
        render_scale = 1.0
        render_w, render_h = orig_w, orig_h
        canvas = img_rgb.copy()

    draw = ImageDraw.Draw(canvas, "RGBA")

    # Scale annotation thickness/size proportionally to canvas size
    scale_factor = max(render_w, render_h) / 640.0
    box_width  = max(1, round(2  * scale_factor))
    tick_len   = max(4, round(12 * scale_factor))
    tick_width = max(1, round(3  * scale_factor))
    dot_r      = max(4, round(10 * scale_factor))
    font_size_b = max(8,  round(15 * scale_factor))
    font_size_r = max(7,  round(12 * scale_factor))
    label_yoff  = max(10, round(22 * scale_factor))

    try:
        fnt_b = ImageFont.truetype(FONT_BOLD, font_size_b)
        fnt   = ImageFont.truetype(FONT_REG,  font_size_r)
    except OSError:
        fnt_b = fnt = ImageFont.load_default()

    for i, pred in enumerate(predictions):
        # Scale box coords to render canvas
        x  = pred["x"]     * render_scale
        y  = pred["y"]     * render_scale
        bw = pred["width"] * render_scale
        bh = pred["height"]* render_scale
        x1, y1 = int(x - bw / 2), int(y - bh / 2)
        x2, y2 = int(x + bw / 2), int(y + bh / 2)

        cls  = pred.get("class", "unknown")
        conf = pred.get("confidence", 0) * 100
        meta = get_meta(cls)
        col  = meta["color"]
        r, g, b_c = _hex_to_rgb(col)

        # Filled box
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b_c, 22), outline=col, width=box_width)

        # Corner ticks
        t = tick_len
        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            draw.line([(px, py + dy*t), (px, py), (px + dx*t, py)], fill=col, width=tick_width)

        # Index dot
        draw.ellipse([x1+2, y1+2, x1+2+dot_r*2, y1+2+dot_r*2], fill=(r, g, b_c, 200))
        draw.text((x1 + dot_r // 2 + 2, y1 + 3), str(i+1), fill="white", font=fnt)

        # Label — drawn above the box; if there's no room above, draw below
        label = f" {cls.upper()}  {conf:.0f}% "
        lx, ly = x1, y1 - label_yoff
        if ly < 0:
            ly = y2 + 2
        try:
            bb = draw.textbbox((lx, ly), label, font=fnt_b)
        except AttributeError:
            bb = (lx, ly, lx + len(label) * font_size_b // 2, ly + font_size_b + 4)
        draw.rectangle([bb[0]-1, bb[1]-1, bb[2]+1, bb[3]+1], fill=(r, g, b_c, 190))
        draw.text((lx, ly), label, fill="white", font=fnt_b)

    # --- Downscale back to original size if we upscaled the canvas ────────────
    if render_scale != 1.0:
        canvas = canvas.resize((orig_w, orig_h), Image.LANCZOS)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE BAR
# ─────────────────────────────────────────────────────────────────────────────
def conf_bar(conf_pct: float, color: str) -> str:
    w = int(conf_pct)
    return f"""
    <div style='background:rgba(255,255,255,.06); border-radius:4px; height:8px;
                width:100%; margin-top:4px;'>
        <div style='background:{color}; width:{w}%; height:8px; border-radius:4px;
                    box-shadow:0 0 6px {color}88;'></div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION CARDS
# ─────────────────────────────────────────────────────────────────────────────
def render_detection_cards(preds: List[dict]) -> None:
    """Render one card per detected object — class name, confidence bar, location."""
    if not preds:
        st.markdown("""
        <div class='card' style='text-align:center; padding:36px;'>
            <div style='font-family:IBM Plex Mono,monospace; color:rgba(200,200,200,0.5); font-size:1.4rem;'>✓</div>
            <div style='font-family:IBM Plex Mono,monospace; color:#888888; font-size:.8rem;
                        letter-spacing:3px; margin-top:8px;'>NO OBJECTS DETECTED</div>
            <div style='color:#606060; font-size:.78rem; margin-top:6px;'>
                Subsurface scan clear above confidence threshold</div>
        </div>""", unsafe_allow_html=True)
        return

    # Sort by confidence (highest first)
    sorted_preds = sorted(preds, key=lambda p: p.get("confidence", 0), reverse=True)

    for i, p in enumerate(sorted_preds):
        cls  = p.get("class", "unknown")
        conf = p.get("confidence", 0) * 100
        meta = get_meta(cls)
        col  = meta["color"]
        icon = meta["icon"]
        cx, cy = p.get("x", 0), p.get("y", 0)
        w_, h_ = p.get("width", 0), p.get("height", 0)
        r, g, b = _hex_to_rgb(col)

        st.markdown(f"""
        <div class='card' style='margin-bottom:10px;
             border-left:3px solid {col};'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='display:flex; align-items:center; gap:10px;'>
                    <span style='font-size:1.3rem;'>{icon}</span>
                    <div>
                        <div style='font-weight:700; color:#e8e8e8; font-size:.95rem;
                                    letter-spacing:.5px;'>{cls.upper()}</div>
                        <div style='font-family:IBM Plex Mono,monospace; font-size:.65rem;
                                    color:#888888; letter-spacing:1px;'>OBJECT #{i+1}</div>
                    </div>
                </div>
                <span style='font-family:IBM Plex Mono,monospace; font-size:.75rem;
                             color:{col}; font-weight:700;'>{conf:.1f}%</span>
            </div>
            <div style='margin-top:10px;'>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='font-family:IBM Plex Mono,monospace; font-size:.7rem; color:#888888;'>CONFIDENCE</span>
                    <span style='font-family:IBM Plex Mono,monospace; font-size:.7rem; color:{col};'>{conf:.1f}%</span>
                </div>
                {conf_bar(conf, col)}
            </div>
            <div style='display:flex; gap:18px; margin-top:10px;'>
                <div style='font-family:IBM Plex Mono,monospace; font-size:.68rem; color:#666666;'>
                    CENTER &nbsp;<span style='color:#b0b0b0;'>({cx:.0f}, {cy:.0f}) px</span>
                </div>
                <div style='font-family:IBM Plex Mono,monospace; font-size:.68rem; color:#666666;'>
                    SIZE &nbsp;<span style='color:#b0b0b0;'>{w_:.0f} × {h_:.0f} px</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def preds_to_csv(preds: List[dict], filename: str = "scan") -> bytes:
    """Serialise prediction list to CSV bytes."""
    fieldnames = ["#", "File", "Class", "Confidence_%",
                  "Center_X", "Center_Y", "Width", "Height"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for i, p in enumerate(preds):
        cls = p.get("class", "—")
        writer.writerow({
            "#":             i + 1,
            "File":          filename,
            "Class":         cls,
            "Confidence_%":  f"{p.get('confidence', 0) * 100:.1f}",
            "Center_X":      f"{p.get('x', 0):.0f}",
            "Center_Y":      f"{p.get('y', 0):.0f}",
            "Width":         f"{p.get('width', 0):.0f}",
            "Height":        f"{p.get('height', 0):.0f}",
        })
    return buf.getvalue().encode()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── User info + logout ────────────────────────────────────────────────────
    _role_badge_color = "#d8d8d8" if _current_role == "admin" else "#ffd700"
    _role_label       = "ADMINISTRATOR" if _current_role == "admin" else "OPERATOR"
    st.markdown(f"""
    <div style='font-family:IBM Plex Mono,monospace; font-size:.72rem;
                color:#666666; padding:10px 0 6px; border-bottom:1px solid rgba(220,220,220,.1);
                margin-bottom:10px;'>
        <span style='color:{_role_badge_color}; font-size:.78rem;'>● </span>
        {_auth_name}<br>
        <span style='color:{_role_badge_color}; font-size:.65rem; letter-spacing:3px;'>
        {_role_label}</span>
        &nbsp;·&nbsp;
        <span style='color:#666666; font-size:.65rem;'>{_auth_username}</span>
    </div>""", unsafe_allow_html=True)

    if _auth_status:
        try:
            _authenticator.logout("⏻  LOGOUT", location="sidebar")
        except Exception:
            pass
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace; font-size:.85rem;
                color:#c0c0c0; letter-spacing:3px; padding:14px 0 10px;'>
    ⬡ AVNL-OFMK AI-Engine · AI-Engine for Buried Object Detection 
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>Detection Parameters</div>", unsafe_allow_html=True)
    confidence = st.slider("Confidence Threshold (%)", 10, 90, 35, 5,
                           help="Minimum confidence score to show a detection. "
                                "Lower values catch faint hyperbolas but may increase false positives.")
    overlap    = st.slider("NMS Overlap Threshold (%)", 10, 90, 30, 5,
                           help="Maximum allowed bounding-box overlap (NMS). "
                                "Lower values suppress more duplicates.")

    st.markdown("---")
    st.markdown("<div class='sec-label'>Tiled Inference</div>", unsafe_allow_html=True)
    use_tiles  = st.checkbox("Enable tiled inference for large images", value=True,
                             help="Splits images wider/taller than the threshold into "
                                  "overlapping tiles so every hyperbola is seen at native resolution.")
    tile_size_ui  = st.select_slider("Tile size (px)", options=[320, 416, 512, 640, 800, 1024],
                                     value=640,
                                     help="Side length of each tile sent to the model. "
                                          "Smaller tiles = finer detail but more API calls.")
    tile_overlap_ui = st.slider("Tile overlap (px)", 32, 256, 128, 32,
                                help="Overlap between adjacent tiles. "
                                     "Larger overlap ensures hyperbolas on tile edges are not missed.")

    st.markdown("---")
    st.markdown("<div class='sec-label'>Small-Image Enhancement</div>", unsafe_allow_html=True)
    use_multiscale = st.checkbox("Multi-scale inference for small images", value=True,
                                 help="For images whose longest edge < 320 px, also runs inference "
                                      "at ×2 and ×3 magnification and merges results with NMS. "
                                      "Strongly recommended for 256×64 and similar GPR strips.")
    use_pad_square = st.checkbox("Pad narrow images to square before inference", value=True,
                                 help="Adds reflected-edge padding on the short axis so the model's "
                                      "receptive field is not starved. Helpful for very wide/flat scans.")
    st.markdown(f"""
    <div style='font-family:IBM Plex Mono,monospace; font-size:.68rem; color:#666666; line-height:1.8;'>
    UPSCALE TARGET &nbsp; {TARGET_INFER_SIZE} px longest edge<br>
    MULTI-SCALE &nbsp;&nbsp;&nbsp;&nbsp; ×1 / ×2 / ×3 (if &lt; {MIN_EDGE_FOR_MULTISCALE} px)
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='sec-label'>Display Options</div>", unsafe_allow_html=True)
    show_cards = st.checkbox("Detection cards",   value=True)
    show_table = st.checkbox("Detection table",   value=True)
    show_json  = st.checkbox("Raw JSON response", value=False)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:IBM Plex Mono,monospace; font-size:.7rem;
                color:#666666; line-height:2;'>
    MODEL &nbsp;&nbsp;&nbsp;&nbsp; {MODEL_ID}<br>
    PROVIDER &nbsp; Roboflow<br>
    TYPE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; YOLOv8 Object Detection<br>
    TILE SIZE &nbsp; {tile_size_ui} px<br>
    OVERLAP &nbsp;&nbsp; {tile_overlap_ui} px<br>
    MULTISCALE &nbsp;{"ON" if use_multiscale else "OFF"}<br>
    SESSION &nbsp;&nbsp; {datetime.now().strftime('%Y-%m-%d')}
    </div>""", unsafe_allow_html=True)

    # ── Admin-only panel ──────────────────────────────────────────────────────
    if _current_role == "admin":
        st.markdown("---")
        st.markdown("<div class='sec-label' style='color:#ffd700; letter-spacing:3px;'>⚙ Admin Panel</div>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-family:IBM Plex Mono,monospace; font-size:.68rem;
                    color:#888888; line-height:2;'>
        API KEY &nbsp;&nbsp; <span style='color:#ffd70099;'>{ROBOFLOW_API_KEY[:6]}{'·'*8}</span><br>
        MODEL &nbsp;&nbsp;&nbsp;&nbsp; {MODEL_ID}<br>
        USERS &nbsp;&nbsp;&nbsp;&nbsp; GPRAdmin · GPRUser · RoboGPR
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
st.markdown(f"""
<div class='gpr-header'>
    <div class='gpr-logotype'>AVNL-OFMK</div>
    <div>
        <div class='gpr-head-title'> AI-Engine for Buried Object Detection </div>
        <div class='gpr-head-sub'>
            <span class='dot-live'></span>
            {now}
        </div>
    </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION KPIs
# ─────────────────────────────────────────────────────────────────────────────
history = st.session_state.scan_history
k1, k2, k3, k4 = st.columns(4)
total_dets = sum(len(r["preds"]) for r in history)
all_classes = [p.get("class","") for r in history for p in r["preds"]]
unique_cls  = len(set(all_classes))

k1.metric("SCANS THIS SESSION", st.session_state.total_scans)
k2.metric("TOTAL DETECTIONS",   total_dets)
k3.metric("UNIQUE CLASSES",     unique_cls)
k4.metric("LAST SCAN",          history[-1]["time"] if history else "—")
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_history, tab_guide = st.tabs([
    "🛰  SINGLE SCAN", "📂  BATCH ANALYSIS", "📋  SCAN HISTORY", "ℹ  GUIDE"
])

# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — Single Scan  (with integrated SGY preprocessing)
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        # ── Accept both SGY (raw) and image files ────────────────────────────
        st.markdown("<div class='sec-label'>Upload GPR Scan</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:IBM Plex Mono,monospace; font-size:.68rem; color:#666666;
                    margin-bottom:8px;'>
        Upload a <b style='color:#b0b0b0'>SEG-Y (.sgy)</b> raw file — it will be preprocessed
        automatically before inference — or a ready-made <b style='color:#b0b0b0'>JPEG / PNG / TIFF</b>
        B-scan image.
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload scan",
            type=["sgy", "SGY", "segy", "SEGY", "png", "jpg", "jpeg", "bmp", "tiff"],
            label_visibility="collapsed",
        )

        # ── SGY preprocessing config (collapsed by default) ──────────────────
        _is_sgy = uploaded is not None and uploaded.name.lower().endswith((".sgy", ".segy"))

        # ── Raw scan preview — above config expander for ALL file types ─────────
        if uploaded is not None:
            if _is_sgy:
                try:
                    _raw_data, _raw_dt, _raw_meta = pp_read_sgy(
                        uploaded.getvalue(), uploaded.name)
                    _raw_preview_pil = pp_plot_bscan(
                        _raw_data,
                        f"Raw B-scan — {_raw_data.shape[0]} × {_raw_data.shape[1]}",
                        cmap="gray",
                    )
                    st.image(_raw_preview_pil,
                             caption=f"📡 Raw B-scan input  ({uploaded.name})",
                             use_container_width=True)
                except Exception:
                    pass
            else:
                _img_preview = Image.open(uploaded)
                _disp_w = min(_img_preview.size[0], 700)
                st.image(_img_preview,
                         caption=f"📡 Raw B-scan input  ({_img_preview.size[0]}×{_img_preview.size[1]} px)",
                         width=_disp_w)

        with st.expander("⚙  SGY Preprocessing Config" + (" — active" if _is_sgy else ""),
                         expanded=_is_sgy):
            # ── Filtering applied automatically in backend (not shown in GUI) ──
            # Dewow: ON, window=39 | Bandpass: ON, 100-900 MHz, Butterworth order=4
            pp_dewow_ui    = True
            pp_dew_win     = 39
            pp_bandpass_ui = True
            pp_bp_lo       = 100.0
            pp_bp_hi       = 900.0
            pp_bp_ord      = 4

            pp_col1, pp_col3 = st.columns(2, gap="small")

            with pp_col1:
                st.markdown("<div class='sec-label'>Gain</div>", unsafe_allow_html=True)
                pp_gain_mode = st.selectbox(
                    "Gain Mode",
                    options=["quadratic", "linear", "agc"],
                    index=0,
                    help="quadratic: best for most GPR  |  linear: uniform  |  agc: AGC",
                )
                pp_gain_preset = st.radio(
                    "Max Gain Level",
                    options=["Min", "Medium", "Max"],
                    index=1,
                    horizontal=True,
                    help="Min: 5 dB  |  Medium: 30 dB  |  Max: 60 dB"
                )
                # Map preset to dB value
                gain_preset_map = {"Min": 5.0, "Medium": 30.0, "Max": 60.0}
                pp_gain_db = gain_preset_map[pp_gain_preset]
                
                pp_agc_win = st.slider("AGC Half-window", 5, 80, 20, 5,
                                       help="Only used when Gain Mode = agc.")

            with pp_col3:
                st.markdown("<div class='sec-label'>Background / Output</div>", unsafe_allow_html=True)
                pp_bg_mode    = st.radio("BG Removal", ["mean", "median"])
                # Trace Normalisation (RMS) — kept ACTIVE but NOT displayed in GUI
                pp_trace_norm = True
                pp_cmap       = "gray"  # locked — always greyscale

        _pp_cfg = {
            "gain_mode":       pp_gain_mode,
            "gain_db":         pp_gain_db,
            "agc_window":      pp_agc_win,
            "bg_mode":         pp_bg_mode,
            "apply_dewow":     pp_dewow_ui,
            "dewow_window":    pp_dew_win,
            "apply_bandpass":  pp_bandpass_ui,
            "bp_low_MHz":      pp_bp_lo,
            "bp_high_MHz":     pp_bp_hi,
            "bp_order":        int(pp_bp_ord),
            "trace_normalise": pp_trace_norm,
            "resize_shape":    (640, 640),
            "jpeg_quality":    95,
            "cmap":            pp_cmap,
        }

        # ── Preview and run button ────────────────────────────────────────────
        img        = None   # PIL image fed to inference
        _pp_result = None   # preprocessing result dict (SGY path only)

        if uploaded:
            if _is_sgy:
                # Build a cache key from file content + relevant cfg params so that
                # changing Gain Level / Gain Mode / BG Removal triggers a fresh run.
                import hashlib as _hl
                _cfg_sig = (
                    _pp_cfg["gain_mode"], _pp_cfg["gain_db"],
                    _pp_cfg["agc_window"], _pp_cfg["bg_mode"],
                    _pp_cfg["trace_normalise"],
                )
                _file_hash = _hl.md5(uploaded.getvalue()).hexdigest()
                _run_key   = (_file_hash,) + _cfg_sig

                # Only re-run the pipeline when the key changes
                if st.session_state.get("_pp_run_key") != _run_key:
                    with st.spinner(f"⚙ Preprocessing {uploaded.name}…"):
                        _pp_result = pp_run_pipeline(
                            uploaded.getvalue(), uploaded.name, _pp_cfg)
                    st.session_state["_pp_run_key"]    = _run_key
                    st.session_state["_pp_run_result"] = _pp_result
                else:
                    _pp_result = st.session_state.get("_pp_run_result")

                if _pp_result["status"] != "OK":
                    st.error(f"❌ Preprocessing failed for {uploaded.name}")
                    st.code("\n".join(_pp_result["log"]), language="text")
                    run_btn = False
                else:
                    img = _pp_result["output_pil"]
                    # Show preprocessing log in collapsed expander
                    with st.expander("📋 Preprocessing log", expanded=False):
                        st.code("\n".join(_pp_result["log"]), language="text")

                    st.success("✅ Preprocessing complete — ready for inference.")
                    st.markdown(f"""
                    <div class='card' style='margin-top:12px;'>
                        <div class='sec-label'>File Metadata</div>
                        <div class='mono' style='font-size:.78rem; color:#90c8a8; line-height:1.9;'>
                            📄 {uploaded.name}<br>
                            📐 {_pp_result.get('n_samples','?')} samples × {_pp_result.get('n_traces','?')} traces<br>
                            🖼 Output: {img.size[0]} × {img.size[1]} px · JPEG<br>
                            💾 {len(uploaded.getvalue())/1024:.1f} KB (raw SGY)<br>
                            🕒 {datetime.now().strftime('%H:%M:%S')}
                        </div>
                    </div>""", unsafe_allow_html=True)
                    # Download processed JPEG
                    st.download_button(
                        "⬇  Download Processed JPEG",
                        data=_pp_result["output_jpeg_bytes"],
                        file_name=f"{Path(uploaded.name).stem}.jpg",
                        mime="image/jpeg",
                        use_container_width=True,
                    )
                    run_btn = st.button("🔍  RUN INFERENCE", use_container_width=True)
            else:
                # Regular image upload — preview already shown above expander
                img = Image.open(uploaded)
                st.markdown(f"""
                <div class='card' style='margin-top:12px;'>
                    <div class='sec-label'>File Metadata</div>
                    <div class='mono' style='font-size:.78rem; color:#90c8a8; line-height:1.9;'>
                        📄 {uploaded.name}<br>
                        📐 {img.size[0]} × {img.size[1]} px · {img.mode}<br>
                        💾 {len(uploaded.getvalue())/1024:.1f} KB<br>
                        🕒 {datetime.now().strftime('%H:%M:%S')}
                    </div>
                </div>""", unsafe_allow_html=True)
                run_btn = st.button("🔍  RUN INFERENCE", use_container_width=True)
        else:
            st.markdown("""
            <div class='empty-state'>
                <div style='font-size:2rem; color:rgba(200,200,200,.2); margin-bottom:12px;'>📡</div>
                <div style='font-family:IBM Plex Mono,monospace; color:rgba(200,200,200,.3);
                            font-size:.75rem; letter-spacing:3px;'>
                    NO SCAN LOADED<br>
                    <span style='font-size:.62rem; color:rgba(200,200,200,.18);'>
                    Upload a .sgy raw file or a GPR B-scan image (PNG / JPEG / TIFF)
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)
            run_btn = False

    with col_right:
        st.markdown("<div class='sec-label'>Detection Output</div>", unsafe_allow_html=True)

        if run_btn and uploaded:
            t0 = time.time()
            with st.spinner("Transmitting to inference engine…"):
                try:
                    result  = run_inference(img, confidence, overlap,
                                            tile=use_tiles,
                                            tile_px=tile_size_ui,
                                            tile_ov=tile_overlap_ui,
                                            multi_scale=use_multiscale,
                                            pad_square=use_pad_square)
                    preds   = result.get("predictions", [])
                    elapsed = time.time() - t0

                    st.session_state.last_preds = preds
                    st.session_state.last_image = img
                    st.session_state.total_scans += 1
                    st.session_state.scan_history.append({
                        "id":        st.session_state.total_scans,
                        "file":      uploaded.name,
                        "time":      datetime.now().strftime("%H:%M:%S"),
                        "preds":     preds,
                        "size":      f"{img.size[0]}×{img.size[1]}",
                        "ms":        f"{elapsed*1000:.0f}ms",
                        "image":     img,          # stored for history preview
                    })

                    annotated = draw_detections(img, preds)
                    _disp_w = min(img.size[0], 700)
                    st.image(annotated,
                             caption=f"🎯 Annotated output  ({annotated.size[0]}×{annotated.size[1]} px)",
                             width=_disp_w)

                    # ── per-scan metrics ──────────────────────────────────
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    unique_in_scan = sorted(set(p.get("class","") for p in preds))
                    avg_conf = (sum(p.get("confidence",0) for p in preds)/len(preds)*100) if preds else 0
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Objects Detected", len(preds))
                    m2.metric("Avg Confidence",   f"{avg_conf:.1f}%")
                    m3.metric("Classes Found",     len(unique_in_scan))

                    W_img, H_img = img.size
                    longest_img  = max(W_img, H_img)
                    if longest_img < TARGET_INFER_SIZE:
                        up_factor = TARGET_INFER_SIZE / longest_img
                        if use_multiscale and longest_img < MIN_EDGE_FOR_MULTISCALE:
                            tile_info = f"  ·  upscaled ×{up_factor:.1f} + multi-scale ×1/×2/×3"
                        else:
                            tile_info = f"  ·  upscaled ×{up_factor:.1f} for inference"
                    elif use_tiles and longest_img > TILE_THRESHOLD:
                        tiled_mode = True
                        stride_ui  = tile_size_ui - tile_overlap_ui
                        n_tiles    = (
                            len(list(range(0, H_img, stride_ui))) *
                            len(list(range(0, W_img, stride_ui)))
                        )
                        tile_info = f"  ·  {n_tiles} tiles @ {tile_size_ui}px"
                    else:
                        tile_info = "  ·  full-image"
                    st.markdown(f"""
                    <div style='font-family:IBM Plex Mono,monospace; font-size:.68rem;
                                color:#666666; text-align:right; margin-top:4px;'>
                        ⚡ Inference completed in {elapsed*1000:.0f} ms{tile_info}
                    </div>""", unsafe_allow_html=True)

                    # ── download buttons ──────────────────────────────────
                    dl1, dl2 = st.columns(2)
                    dl1.download_button(
                        "⬇  Download Annotated Image",
                        data=image_to_bytes(annotated),
                        file_name=f"gpr_annotated_{uploaded.name.rsplit('.',1)[0]}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                    dl2.download_button(
                        "⬇  Download CSV Report",
                        data=preds_to_csv(preds, uploaded.name),
                        file_name=f"gpr_report_{uploaded.name.rsplit('.',1)[0]}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    # ── detection cards ───────────────────────────────────
                    if show_cards:
                        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                        st.markdown("<div class='sec-label'>Object Details</div>",
                                    unsafe_allow_html=True)
                        render_detection_cards(preds)

                    # ── detection table ───────────────────────────────────
                    if show_table and preds:
                        st.markdown("<div class='sec-label' style='margin-top:10px;'>Detection Table</div>",
                                    unsafe_allow_html=True)

                        def _conf_color(pct: float) -> str:
                            if pct >= 70: return "🟢"
                            if pct >= 45: return "🟡"
                            return "🔴"

                        rows = [{
                            "#":          i + 1,
                            "Class":      p.get("class", "—"),
                            "Confidence": f"{_conf_color(p.get('confidence',0)*100)} "
                                          f"{p.get('confidence',0)*100:.1f}%",
                            "Center X":   f"{p.get('x',0):.0f}",
                            "Center Y":   f"{p.get('y',0):.0f}",
                            "Width":      f"{p.get('width',0):.0f}",
                            "Height":     f"{p.get('height',0):.0f}",
                        } for i, p in enumerate(preds)]
                        st.dataframe(rows, use_container_width=True, hide_index=True)

                    if show_json:
                        with st.expander("📄 Raw JSON Response"):
                            st.code(json.dumps(result, indent=2), language="json")

                except requests.exceptions.HTTPError as e:
                    st.error(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected inference error: {type(e).__name__}: {e}")

        elif st.session_state.last_preds and st.session_state.last_image:
            annotated = draw_detections(st.session_state.last_image,
                                        st.session_state.last_preds)
            _disp_w = min(st.session_state.last_image.size[0], 700)
            st.image(annotated, caption="🎯 Last detection result", width=_disp_w)
            if show_cards:
                render_detection_cards(st.session_state.last_preds)
        else:
            st.markdown("""
            <div class='empty-state'>
                <div style='font-size:2rem; color:rgba(200,200,200,.15); margin-bottom:12px;'>🎯</div>
                <div style='font-family:IBM Plex Mono,monospace; color:rgba(200,200,200,.28);
                            font-size:.75rem; letter-spacing:3px;'>AWAITING SCAN INPUT</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("<div class='sec-label'>Batch B-scan Processing</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>Upload multiple GPR B-scan images. The system runs inference on each
    sequentially and produces an aggregated threat report with per-file annotated previews.</div>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload scans", type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if batch_files:
        st.info(f"📂 {len(batch_files)} scan(s) queued for processing")
        if st.button("▶  PROCESS ALL SCANS", use_container_width=False):
            prog    = st.progress(0, text="Starting…")
            results = []
            batch_detail: List[dict] = []   # stores annotated images for preview

            for i, f in enumerate(batch_files):
                prog.progress(i / len(batch_files),
                              text=f"Processing {f.name}  [{i+1}/{len(batch_files)}]")
                try:
                    img_b = Image.open(f)
                    res   = run_inference(img_b, confidence, overlap,
                                         tile=use_tiles,
                                         tile_px=tile_size_ui,
                                         tile_ov=tile_overlap_ui,
                                         multi_scale=use_multiscale,
                                         pad_square=use_pad_square)
                    preds = res.get("predictions", [])
                    ann   = draw_detections(img_b, preds)
                    avg_c = (sum(p.get("confidence",0) for p in preds)/len(preds)*100) if preds else 0.0
                    results.append({
                        "File":        f.name,
                        "Detections":  len(preds),
                        "Avg Conf %":  f"{avg_c:.1f}",
                        "Classes":     ", ".join(sorted(set(p.get("class","") for p in preds))) or "—",
                        "Status":      "✅ OK",
                    })
                    batch_detail.append({"name": f.name, "image": ann, "preds": preds})
                    st.session_state.total_scans += 1
                    st.session_state.scan_history.append({
                        "id":    st.session_state.total_scans,
                        "file":  f.name,
                        "time":  datetime.now().strftime("%H:%M:%S"),
                        "preds": preds,
                        "size":  f"{img_b.size[0]}×{img_b.size[1]}",
                        "ms":    "—",
                        "image": img_b,
                    })
                except Exception as e:
                    results.append({
                        "File":       f.name,
                        "Detections": "ERR",
                        "Avg Conf %": "—",
                        "Classes":    f"{type(e).__name__}: {str(e)[:60]}",
                        "Status":     "❌",
                    })
                    batch_detail.append({"name": f.name, "image": None, "preds": []})
                time.sleep(0.15)

            prog.progress(1.0, text="✅ Batch complete")

            # ── summary table ─────────────────────────────────────────────
            st.markdown("<div class='sec-label' style='margin-top:14px;'>Batch Report</div>",
                        unsafe_allow_html=True)
            st.dataframe(results, use_container_width=True, hide_index=True)

            valid = [r for r in results if isinstance(r["Detections"], int)]
            if valid:
                b1, b2, b3 = st.columns(3)
                b1.metric("Scans Processed", len(batch_files))
                b2.metric("Total Objects",   sum(r["Detections"] for r in valid))
                b3.metric("Clean Scans",     sum(1 for r in valid if r["Detections"] == 0))

            # ── CSV download for the whole batch ─────────────────────────
            all_preds_flat = []
            for detail in batch_detail:
                for p in detail["preds"]:
                    p_copy = dict(p)
                    p_copy["_file"] = detail["name"]
                    all_preds_flat.append(p_copy)

            if all_preds_flat:
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=[
                    "File","Class","Confidence_%",
                    "Center_X","Center_Y","Width","Height"])
                writer.writeheader()
                for p in all_preds_flat:
                    cls = p.get("class","—")
                    writer.writerow({
                        "File":          p.get("_file",""),
                        "Class":         cls,
                        "Confidence_%":  f"{p.get('confidence',0)*100:.1f}",
                        "Center_X":      f"{p.get('x',0):.0f}",
                        "Center_Y":      f"{p.get('y',0):.0f}",
                        "Width":         f"{p.get('width',0):.0f}",
                        "Height":        f"{p.get('height',0):.0f}",
                    })
                st.download_button(
                    "⬇  Download Full Batch CSV Report",
                    data=buf.getvalue().encode(),
                    file_name="gpr_batch_report.csv",
                    mime="text/csv",
                )

            # ── per-file annotated previews ───────────────────────────────
            if batch_detail:
                st.markdown("<div class='sec-label' style='margin-top:16px;'>Per-file Previews</div>",
                            unsafe_allow_html=True)
                for detail in batch_detail:
                    label = (f"📄 {detail['name']}  —  "
                             f"{len(detail['preds'])} detection(s)")
                    with st.expander(label, expanded=False):
                        if detail["image"] is not None:
                            st.image(detail["image"], use_container_width=True)
                            if detail["preds"]:
                                render_detection_cards(detail["preds"])
                        else:
                            st.error("Inference failed for this file.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — History
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("<div class='sec-label'>Session Scan History</div>", unsafe_allow_html=True)

    if not st.session_state.scan_history:
        st.markdown("""
        <div class='empty-state'>
            <div style='font-family:IBM Plex Mono,monospace; color:#606060;
                        font-size:.8rem; letter-spacing:3px;'>NO SCANS LOGGED YET</div>
        </div>""", unsafe_allow_html=True)
    else:
        rows = []
        for r in reversed(st.session_state.scan_history):
            preds = r["preds"]
            avg_c = (sum(p.get("confidence",0) for p in preds)/len(preds)*100) if preds else 0.0
            rows.append({
                "Scan #":     r["id"],
                "File":       r["file"],
                "Time":       r["time"],
                "Size":       r["size"],
                "Detections": len(preds),
                "Avg Conf %": f"{avg_c:.1f}",
                "Classes":    ", ".join(sorted(set(p.get("class","") for p in preds))) or "—",
                "Latency":    r.get("ms","—"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # ── per-scan expandable preview ───────────────────────────────────
        st.markdown("<div class='sec-label' style='margin-top:14px;'>Scan Previews</div>",
                    unsafe_allow_html=True)
        for r in reversed(st.session_state.scan_history):
            if r.get("image") is None:
                continue
            _avg = (sum(p.get("confidence",0) for p in r["preds"])/len(r["preds"])*100) if r["preds"] else 0
            label = f"Scan #{r['id']}  ·  {r['file']}  ·  {r['time']}  ·  {len(r['preds'])} object(s)  ·  avg {_avg:.0f}% conf"
            with st.expander(label, expanded=False):
                ann = draw_detections(r["image"], r["preds"])
                st.image(ann, use_container_width=True)
                if r["preds"]:
                    render_detection_cards(r["preds"])

        if st.button("🗑  Clear History"):
            st.session_state.scan_history = []
            st.session_state.total_scans  = 0
            st.session_state.last_preds   = []
            st.session_state.last_image   = None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Guide
# ══════════════════════════════════════════════════════════════════════════════
with tab_guide:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class='card card-lo'>
            <div class='sec-label'>How to Use</div>
            <div style='font-size:.88rem; color:#b0b0b0; line-height:1.9;'>
            1. Go to the <b style='color:#e0e0e0'>Single Scan</b> tab<br>
            2. Upload a <b style='color:#e0e0e0'>SEG-Y (.sgy)</b> raw file <em>or</em> a GPR B-scan image (PNG / JPEG / TIFF)<br>
            3. If SGY: the pipeline preprocesses automatically — adjust parameters in the expander first<br>
            4. Adjust <b style='color:#e0e0e0'>Confidence</b> &amp; <b style='color:#e0e0e0'>Overlap</b> in the sidebar<br>
            5. Click <b style='color:#e0e0e0'>RUN INFERENCE</b><br>
            6. Review the annotated output, detection cards &amp; table<br>
            7. Use <b style='color:#e0e0e0'>⬇ Download</b> buttons to export image or CSV<br>
            8. Use <b style='color:#e0e0e0'>Batch Analysis</b> for multiple scans at once
            </div>
        </div>
        <div class='card'>
            <div class='sec-label'>Detected Object Classes</div>
            <div class='mono' style='font-size:.78rem; color:#b0b0b0; line-height:2.1;'>
            <span style='color:#ff3030'>■</span> Landmine &nbsp; <span style='color:#ff5555'>■</span> Mine &nbsp; <span style='color:#ff2020'>■</span> IED<br>
            <span style='color:#ff6600'>■</span> Threat &nbsp;&nbsp;&nbsp; <span style='color:#ff8c00'>■</span> Metal &nbsp; <span style='color:#ffa500'>■</span> Pipe<br>
            <span style='color:#ffd700'>■</span> Cable &nbsp;&nbsp;&nbsp;&nbsp; <span style='color:#ffe066'>■</span> Utility<br>
            <span style='color:#00ffb4'>■</span> Rock &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style='color:#7ac9a9'>■</span> Root<br>
            <span style='color:#00bfff'>■</span> Void &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style='color:#aaaaaa'>■</span> Clutter
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='card card-lo'>
            <div class='sec-label'>About GPR B-scans</div>
            <div style='font-size:.88rem; color:#b0b0b0; line-height:1.85;'>
            Ground Penetrating Radar B-scans are 2-D cross-sectional profiles of subsurface
            reflectivity. Buried objects appear as characteristic
            <b style='color:#e0e0e0'>hyperbolic reflections</b> whose apex depth and curvature
            encode the object's depth and the soil's dielectric constant.<br><br>
            This platform uses a YOLOv8 model trained on real GPR data to detect and classify
            these signatures across varying soil conditions.
            </div>
        </div>
        <div class='card'>
            <div class='sec-label'>Recommended Settings</div>
            <div class='mono' style='font-size:.78rem; color:#b0b0b0; line-height:2.1;'>
            High-clutter soil &nbsp;&nbsp;&nbsp; Confidence ≥ 50%<br>
            Clean / dry soil &nbsp;&nbsp;&nbsp;&nbsp; Confidence ≥ 35%<br>
            Dense object fields &nbsp; Overlap ≤ 25%<br>
            Sparse scenes &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Overlap ≤ 40%
            </div>
        </div>
        <div class='card'>
            <div class='sec-label'>API Key Configuration</div>
            <div style='font-size:.82rem; color:#b0b0b0; line-height:1.85;'>
            Set your Roboflow key in <b style='color:#e0e0e0'>.streamlit/secrets.toml</b>:<br>
            <span class='mono' style='font-size:.75rem; color:#888888;'>
            ROBOFLOW_API_KEY = "your_key_here"</span><br><br>
            Or export as an environment variable before running:<br>
            <span class='mono' style='font-size:.75rem; color:#888888;'>
            export ROBOFLOW_API_KEY=your_key_here</span>
            </div>
        </div>""", unsafe_allow_html=True)




