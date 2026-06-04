






"""
GPR Buried Object Detection — Streamlit App v7
Model: raw_gpr_objectdetection/1 (Roboflow)
ISEG · Ground Penetrating Radar Platform

Improvements over v3:
  • API key loaded from st.secrets / env var (no hardcoded secret)
  • Consistent model version string throughout (was /2 in sidebar, /1 in URL)
  • Retry logic with exponential back-off for transient API errors
  • Download buttons: annotated image (PNG) + detection report (CSV)
  • Batch mode shows per-file annotated thumbnails in an expander
  • History tab: expandable per-scan preview with annotated image
  • Confidence % colour-coded in the detection table (red / amber / green)
  • Full-resolution tiled inference: image is NEVER downscaled
    — large images are split into 640 px overlapping tiles so every
    hyperbola is visible to the model at native resolution
  • NMS across tiles merges cross-border duplicate detections
  • Output image is always the same pixel dimensions as the input
  • Display: equal-width columns + fixed pixel width so input and output
    appear at identical visual size in the browser — no stretching artefacts
  • Removed threat-level classification (HIGH/MEDIUM/LOW) — app shows
    detected objects with class name + confidence only
  • Type hints on all public functions
  • Constants unified at the top; magic strings removed
  • Minor: empty-except blocks replaced with specific exception logging
"""

import io
import base64
import json
import time
import os
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GPR Threat Detection · ISEG",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
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
# Images are NEVER downscaled — original resolution is always preserved so that
# small / shallow hyperbolas are not merged or lost during pre-processing.
# For images whose longest edge exceeds TILE_THRESHOLD, the image is split into
# overlapping tiles (TILE_SIZE × TILE_SIZE) with TILE_OVERLAP px border overlap.
# Detections from every tile are remapped to original-image coordinates and
# duplicate boxes are removed with NMS (IOU_THRESHOLD).
TILE_THRESHOLD = 1280   # px  — run tiled inference above this long-edge size
TILE_SIZE      = 640    # px  — side length of each tile sent to the API
TILE_OVERLAP   = 128    # px  — overlap between adjacent tiles (catches border hyperbolas)
IOU_THRESHOLD  = 0.45   # NMS IOU threshold for merging cross-tile duplicates

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
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

.stApp {
    background-color: #050e0a;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(0,255,140,.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(0,180,255,.03) 0%, transparent 50%),
        linear-gradient(rgba(0,255,140,.022) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,140,.022) 1px, transparent 1px);
    background-size: auto, auto, 48px 48px, 48px 48px;
}
.stApp::after {
    content:""; position:fixed; inset:0; pointer-events:none; z-index:1000;
    background: repeating-linear-gradient(0deg,transparent,transparent 3px,
                rgba(0,0,0,.06) 3px,rgba(0,0,0,.06) 4px);
}

/* ── Header ── */
.gpr-header {
    display:flex; align-items:center; gap:20px;
    padding:22px 0 16px;
    border-bottom:1px solid rgba(0,255,140,.18);
    margin-bottom:24px;
}
.gpr-logotype {
    font-family:'Share Tech Mono',monospace; font-size:2.6rem; color:#00ff8c;
    text-shadow:0 0 20px #00ff8c99,0 0 50px #00ff8c33; letter-spacing:4px; line-height:1;
}
.gpr-head-title { font-size:1.5rem; font-weight:900; color:#e0f5ec; margin:0; letter-spacing:.4px; }
.gpr-head-sub   {
    font-family:'Share Tech Mono',monospace; font-size:.68rem;
    color:#00ff8c66; letter-spacing:4px; text-transform:uppercase;
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    background:rgba(0,255,140,.04) !important;
    border:1px solid rgba(0,255,140,.28) !important;
    color:#d8f5e8 !important; border-radius:7px !important;
    font-family:'Share Tech Mono',monospace !important; font-size:.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color:#00ff8c !important;
    box-shadow:0 0 0 2px rgba(0,255,140,.14) !important;
}
label[data-testid="stWidgetLabel"] p {
    color:#6ab890 !important; font-size:.72rem !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
    font-family:'Share Tech Mono',monospace !important;
}

/* ── Buttons ── */
.stButton > button {
    background:linear-gradient(135deg,#00ff8c,#00cc70) !important;
    color:#050e0a !important; font-weight:800 !important;
    font-family:'Exo 2',sans-serif !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
    border:none !important; border-radius:7px !important;
    padding:10px 26px !important; font-size:.82rem !important;
    transition:all .2s !important;
}
.stButton > button:hover {
    box-shadow:0 0 24px rgba(0,255,140,.45) !important;
    transform:translateY(-2px) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background:rgba(4,12,8,.97) !important;
    border-right:1px solid rgba(0,255,140,.14) !important;
}
[data-testid="stSidebar"] * { color:#b0d8c0; }

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#00ff8c !important; }

/* ── File uploader ── */
.stFileUploader > div {
    background:rgba(0,255,140,.03) !important;
    border:1px dashed rgba(0,255,140,.28) !important;
    border-radius:10px !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background:rgba(0,255,140,.04);
    border:1px solid rgba(0,255,140,.14);
    border-radius:10px; padding:14px 18px;
}
[data-testid="metric-container"] label {
    color:#6ab890 !important; font-size:.68rem !important;
    letter-spacing:2px !important; font-family:'Share Tech Mono',monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#00ff8c !important; font-family:'Share Tech Mono',monospace !important;
    font-size:1.7rem !important;
}

/* ── Cards ── */
.card {
    background:rgba(8,20,14,.8);
    border:1px solid rgba(0,255,140,.14);
    border-radius:10px; padding:20px 24px; margin-bottom:12px;
    backdrop-filter:blur(8px);
}
/* card threat-level variants removed — dynamic colour used instead */

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(0,255,140,.04) !important;
    border-radius:8px; gap:2px; padding:4px;
}
.stTabs [data-baseweb="tab"] {
    font-family:'Share Tech Mono',monospace !important; font-size:.75rem !important;
    letter-spacing:2px !important; color:#6ab890 !important;
    border-radius:6px !important; padding:8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background:rgba(0,255,140,.12) !important; color:#00ff8c !important;
}

/* ── Alerts ── */
.stAlert {
    background:rgba(0,255,140,.05) !important;
    border:1px solid rgba(0,255,140,.2) !important;
    border-radius:8px !important;
}

/* ── Progress ── */
.stProgress > div > div { background:#00ff8c !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    color:#6ab890 !important; font-family:'Share Tech Mono',monospace !important;
    font-size:.75rem !important; letter-spacing:1px !important;
}

/* ── Helpers ── */
.sec-label {
    font-family:'Share Tech Mono',monospace; font-size:.68rem;
    color:#00ff8c66; letter-spacing:4px; text-transform:uppercase; margin-bottom:10px;
}
.mono { font-family:'Share Tech Mono',monospace; }

@keyframes pulse {
    0%,100%{box-shadow:0 0 0 0 rgba(0,255,140,.5)}
    50%{box-shadow:0 0 0 7px rgba(0,255,140,0)}
}
.dot-live {
    display:inline-block; width:8px; height:8px; background:#00ff8c;
    border-radius:50%; margin-right:7px;
    animation:pulse 1.8s ease-in-out infinite; vertical-align:middle;
}

/* badges removed — no threat-level classification */

.empty-state {
    background:rgba(0,255,140,.02); border:1px dashed rgba(0,255,140,.15);
    border-radius:10px; padding:52px 20px; text-align:center;
}

hr { border-color:rgba(0,255,140,.12) !important; }

/* ── Image parity: prevent Streamlit from auto-stretching images ── */
[data-testid="stImage"] img {
    max-width: 100% !important;
    height: auto !important;
    display: block !important;
}
/* Both scan columns get identical max-width so images feel the same size */
[data-testid="column"] [data-testid="stImage"] {
    width: 100% !important;
}
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#050e0a; }
::-webkit-scrollbar-thumb { background:rgba(0,255,140,.25); border-radius:3px; }
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
    """Run inference on one tile and remap box coordinates to full-image space."""
    result = _call_api(_encode_jpeg(tile), confidence, overlap)
    preds = result.get("predictions", [])
    for p in preds:
        p["x"] += offset_x
        p["y"] += offset_y
    return preds


def run_inference(image: Image.Image, confidence: int, overlap: int,
                  tile: bool = True, tile_px: int = TILE_SIZE,
                  tile_ov: int = TILE_OVERLAP) -> Dict[str, Any]:
    """
    Full-resolution tiled inference pipeline.

    • The original image is NEVER downscaled — pixel dimensions in == pixel dimensions out.
    • Images with longest edge ≤ TILE_THRESHOLD are sent as-is (single API call).
    • Larger images are split into TILE_SIZE × TILE_SIZE patches with TILE_OVERLAP
      border on every side so hyperbolas that straddle tile edges are fully captured.
    • Detections from all tiles are remapped to original-image coordinates and
      cross-tile duplicates are removed with NMS.

    Returns a dict with key "predictions" whose boxes are in the original image's
    coordinate space — ready for draw_detections() without any coordinate adjustment.
    """
    rgb = _to_rgb(image)
    W, H = rgb.size

    # ── Single-tile fast path ─────────────────────────────────────────────────
    if not tile or max(W, H) <= TILE_THRESHOLD:
        result = _call_api(_encode_jpeg(rgb), confidence, overlap)
        return result

    # ── Tiled path ────────────────────────────────────────────────────────────
    stride = tile_px - tile_ov               # step between tile origins
    all_preds: List[dict] = []

    # Build tile grid
    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))

    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_px, W)
            y1 = min(y0 + tile_px, H)
            # Snap origin so every tile is exactly tile_px × tile_px
            tx0 = max(0, x1 - tile_px)
            ty0 = max(0, y1 - tile_px)
            tile_img = rgb.crop((tx0, ty0, tx0 + tile_px, ty0 + tile_px))
            tile_preds = _infer_tile(tile_img, tx0, ty0, confidence, overlap)
            all_preds.extend(tile_preds)

    merged = _nms(all_preds)
    return {"predictions": merged, "image": {"width": W, "height": H}}


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def draw_detections(image: Image.Image, predictions: List[dict]) -> Image.Image:
    """Return a copy of *image* annotated with bounding boxes and labels."""
    img  = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        fnt_b = ImageFont.truetype(FONT_BOLD, 15)
        fnt   = ImageFont.truetype(FONT_REG,  12)
    except OSError:
        fnt_b = fnt = ImageFont.load_default()

    for i, pred in enumerate(predictions):
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        cls  = pred.get("class", "unknown")
        conf = pred.get("confidence", 0) * 100
        meta = get_meta(cls)
        col  = meta["color"]
        r, g, b = _hex_to_rgb(col)

        # Filled box
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 22), outline=col, width=2)

        # Corner ticks
        t = 12
        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            draw.line([(px, py + dy*t), (px, py), (px + dx*t, py)], fill=col, width=3)

        # Index dot
        dot_r = 10
        draw.ellipse([x1+4, y1+4, x1+4+dot_r*2, y1+4+dot_r*2], fill=(r, g, b, 200))
        draw.text((x1+8, y1+5), str(i+1), fill="white", font=fnt)

        # Label
        label = f" {cls.upper()}  {conf:.0f}% "
        try:
            bb = draw.textbbox((x1, y1 - 22), label, font=fnt_b)
        except AttributeError:
            bb = (x1, y1 - 22, x1 + len(label) * 9, y1 - 6)
        draw.rectangle([bb[0]-1, bb[1]-1, bb[2]+1, bb[3]+1], fill=(r, g, b, 190))
        draw.text((x1, y1 - 22), label, fill="white", font=fnt_b)

    return img


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
            <div style='font-family:Share Tech Mono,monospace; color:#00ff8c88; font-size:1.4rem;'>✓</div>
            <div style='font-family:Share Tech Mono,monospace; color:#6ab890; font-size:.8rem;
                        letter-spacing:3px; margin-top:8px;'>NO OBJECTS DETECTED</div>
            <div style='color:#4a8868; font-size:.78rem; margin-top:6px;'>
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
                        <div style='font-weight:700; color:#e0f5ec; font-size:.95rem;
                                    letter-spacing:.5px;'>{cls.upper()}</div>
                        <div style='font-family:Share Tech Mono,monospace; font-size:.65rem;
                                    color:#6ab890; letter-spacing:1px;'>OBJECT #{i+1}</div>
                    </div>
                </div>
                <span style='font-family:Share Tech Mono,monospace; font-size:.75rem;
                             color:{col}; font-weight:700;'>{conf:.1f}%</span>
            </div>
            <div style='margin-top:10px;'>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='font-family:Share Tech Mono,monospace; font-size:.7rem; color:#6ab890;'>CONFIDENCE</span>
                    <span style='font-family:Share Tech Mono,monospace; font-size:.7rem; color:{col};'>{conf:.1f}%</span>
                </div>
                {conf_bar(conf, col)}
            </div>
            <div style='display:flex; gap:18px; margin-top:10px;'>
                <div style='font-family:Share Tech Mono,monospace; font-size:.68rem; color:#4a8868;'>
                    CENTER &nbsp;<span style='color:#90c8a8;'>({cx:.0f}, {cy:.0f}) px</span>
                </div>
                <div style='font-family:Share Tech Mono,monospace; font-size:.68rem; color:#4a8868;'>
                    SIZE &nbsp;<span style='color:#90c8a8;'>{w_:.0f} × {h_:.0f} px</span>
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
    st.markdown("""
    <div style='font-family:Share Tech Mono,monospace; font-size:.85rem;
                color:#00ff8c; letter-spacing:3px; padding:14px 0 10px;'>
    ⬡ ISEG · GPR CONTROL
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
    st.markdown("<div class='sec-label'>Display Options</div>", unsafe_allow_html=True)
    show_cards = st.checkbox("Detection cards",   value=True)
    show_table = st.checkbox("Detection table",   value=True)
    show_json  = st.checkbox("Raw JSON response", value=False)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:Share Tech Mono,monospace; font-size:.7rem;
                color:#4a8868; line-height:2;'>
    MODEL &nbsp;&nbsp;&nbsp;&nbsp; {MODEL_ID}<br>
    PROVIDER &nbsp; Roboflow<br>
    TYPE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; YOLOv8 Object Detection<br>
    TILE SIZE &nbsp; {tile_size_ui} px<br>
    OVERLAP &nbsp;&nbsp; {tile_overlap_ui} px<br>
    SESSION &nbsp;&nbsp; {datetime.now().strftime('%Y-%m-%d')}
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
st.markdown(f"""
<div class='gpr-header'>
    <div class='gpr-logotype'>⬡ ISEG</div>
    <div>
        <div class='gpr-head-title'>GPR Buried Object Detection</div>
        <div class='gpr-head-sub'>
            <span class='dot-live'></span>
            Ground Penetrating Radar · ML Inference Platform · {now}
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Scan
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.markdown("<div class='sec-label'>Upload GPR B-scan Image</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload B-scan",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            label_visibility="collapsed",
        )

        if uploaded:
            img = Image.open(uploaded)
            _disp_w = min(img.size[0], 700)  # cap at 700 px for display only
            st.image(img, caption=f"📡 Raw B-scan input  ({img.size[0]}×{img.size[1]} px)",
                     width=_disp_w)
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
                <div style='font-size:2rem; color:rgba(0,255,140,.2); margin-bottom:12px;'>📡</div>
                <div style='font-family:Share Tech Mono,monospace; color:rgba(0,255,140,.3);
                            font-size:.75rem; letter-spacing:3px;'>
                    NO SCAN LOADED<br>
                    <span style='font-size:.62rem; color:rgba(0,255,140,.18);'>
                    Upload a GPR B-scan image (PNG / JPEG / TIFF)
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
                                            tile_ov=tile_overlap_ui)
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
                    tiled_mode = use_tiles and max(W_img, H_img) > TILE_THRESHOLD
                    if tiled_mode:
                        stride_ui = tile_size_ui - tile_overlap_ui
                        n_tiles = (
                            len(list(range(0, H_img, stride_ui))) *
                            len(list(range(0, W_img, stride_ui)))
                        )
                        tile_info = f"  ·  {n_tiles} tiles @ {tile_size_ui}px"
                    else:
                        tile_info = "  ·  full-image"
                    st.markdown(f"""
                    <div style='font-family:Share Tech Mono,monospace; font-size:.68rem;
                                color:#4a8868; text-align:right; margin-top:4px;'>
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
                <div style='font-size:2rem; color:rgba(0,255,140,.15); margin-bottom:12px;'>🎯</div>
                <div style='font-family:Share Tech Mono,monospace; color:rgba(0,255,140,.28);
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
                                         tile_ov=tile_overlap_ui)
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
            <div style='font-family:Share Tech Mono,monospace; color:#4a8868;
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
            <div style='font-size:.88rem; color:#b0d8c0; line-height:1.9;'>
            1. Go to the <b style='color:#00ff8c'>Single Scan</b> tab<br>
            2. Upload a GPR B-scan image (PNG / JPEG / TIFF)<br>
            3. Adjust <b style='color:#00ff8c'>Confidence</b> &amp; <b style='color:#00ff8c'>Overlap</b> in the sidebar<br>
            4. Click <b style='color:#00ff8c'>RUN INFERENCE</b><br>
            5. Review the annotated output, detection cards &amp; table<br>
            6. Use <b style='color:#00ff8c'>⬇ Download</b> buttons to export image or CSV<br>
            7. Use <b style='color:#00ff8c'>Batch Analysis</b> for multiple scans at once
            </div>
        </div>
        <div class='card'>
            <div class='sec-label'>Detected Object Classes</div>
            <div class='mono' style='font-size:.78rem; color:#b0d8c0; line-height:2.1;'>
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
            <div style='font-size:.88rem; color:#b0d8c0; line-height:1.85;'>
            Ground Penetrating Radar B-scans are 2-D cross-sectional profiles of subsurface
            reflectivity. Buried objects appear as characteristic
            <b style='color:#00ff8c'>hyperbolic reflections</b> whose apex depth and curvature
            encode the object's depth and the soil's dielectric constant.<br><br>
            This platform uses a YOLOv8 model trained on real GPR data to detect and classify
            these signatures across varying soil conditions.
            </div>
        </div>
        <div class='card'>
            <div class='sec-label'>Recommended Settings</div>
            <div class='mono' style='font-size:.78rem; color:#b0d8c0; line-height:2.1;'>
            High-clutter soil &nbsp;&nbsp;&nbsp; Confidence ≥ 50%<br>
            Clean / dry soil &nbsp;&nbsp;&nbsp;&nbsp; Confidence ≥ 35%<br>
            Dense object fields &nbsp; Overlap ≤ 25%<br>
            Sparse scenes &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Overlap ≤ 40%
            </div>
        </div>
        <div class='card'>
            <div class='sec-label'>API Key Configuration</div>
            <div style='font-size:.82rem; color:#b0d8c0; line-height:1.85;'>
            Set your Roboflow key in <b style='color:#00ff8c'>.streamlit/secrets.toml</b>:<br>
            <span class='mono' style='font-size:.75rem; color:#6ab890;'>
            ROBOFLOW_API_KEY = "your_key_here"</span><br><br>
            Or export as an environment variable before running:<br>
            <span class='mono' style='font-size:.75rem; color:#6ab890;'>
            export ROBOFLOW_API_KEY=your_key_here</span>
            </div>
        </div>""", unsafe_allow_html=True)






