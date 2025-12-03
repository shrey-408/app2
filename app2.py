# app.py
# Single-file Streamlit app using only OpenCV + numpy + streamlit.
# It will try to pip-install missing packages at runtime if necessary.
# Note: runtime pip installs may not work on all hosting environments (Streamlit Cloud often requires requirements.txt).
import sys
import subprocess
import importlib
import time
import math
from typing import List, Tuple

# -----------------------------
# Auto-install helper (best-effort)
# -----------------------------
def ensure_pkg(pkg_name: str, import_name: str = None, version_spec: str = ""):
    """
    Try to import import_name (or pkg_name). If missing, pip install pkg_name(+version_spec) then import.
    Returns the imported module, or raises ImportError.
    """
    import_name = import_name or pkg_name
    try:
        return importlib.import_module(import_name)
    except Exception:
        pkg_spec = pkg_name + (version_spec if version_spec else "")
        print(f"Package '{import_name}' not found. Attempting to pip install '{pkg_spec}' ...", file=sys.stderr)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_spec])
            time.sleep(0.1)
            return importlib.import_module(import_name)
        except Exception as e:
            raise ImportError(f"Failed to import or install '{pkg_spec}': {e}") from e

# Attempt to ensure required packages are present.
# If you run in an environment that disallows pip installs, this will raise and show an explanation in the UI.
try:
    st = ensure_pkg("streamlit")
    np = ensure_pkg("numpy")
    cv2 = ensure_pkg("opencv-python-headless", import_name="cv2")
except ImportError as e:
    # If installation failed, show a helpful message using minimal stdout fallback.
    print(str(e), file=sys.stderr)
    # Create a very small fallback Streamlit-like interface using plain prints if streamlit import failed.
    # But in practice, running "streamlit run app.py" is recommended so st exists.
    raise

# Now we have st, np, cv2 available
import streamlit as st  # re-import with proper name
import numpy as np
import cv2

# -----------------------------
# App code starts here
# -----------------------------
st.set_page_config(page_title="OpenCV Image Toolbox (single file)", page_icon="🧰", layout="wide")
st.title("OpenCV Image Toolbox — single file, OpenCV-only (attempts runtime install if needed)")

# Utilities
def read_image_from_upload(uploaded_file) -> Tuple[np.ndarray, str]:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError("Could not decode image. Is the file a valid image?")
    return img_bgr, uploaded_file.name

def to_display_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    if img_bgr.shape[2] == 4:
        rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGBA)
        return rgba
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def encode_image_to_bytes(img_bgr: np.ndarray, ext: str = ".png") -> bytes:
    success, buf = cv2.imencode(ext, img_bgr)
    if not success:
        raise ValueError("Failed to encode image.")
    return buf.tobytes()

def is_prime(n: int) -> bool:
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0: return False
    r = int(math.sqrt(n))
    for i in range(3, r+1, 2):
        if n % i == 0:
            return False
    return True

def get_properties(img_bgr: np.ndarray, filename: str) -> dict:
    h, w = img_bgr.shape[:2]
    channels = 1 if img_bgr.ndim == 2 else img_bgr.shape[2]
    dtype = str(img_bgr.dtype)
    return {
        "Filename": filename,
        "Width": int(w),
        "Height": int(h),
        "Channels": int(channels),
        "Dtype": dtype,
        "Num pixels": int(w*h),
        "Aspect ratio (W:H)": f"{w}:{h}",
    }

def simple_object_detection(img_bgr: np.ndarray, min_area: int = 500) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img_bgr.copy() if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        boxes.append((x,y,w,h))
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(out, str(int(area)), (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return out, boxes

def rotate_image(img_bgr: np.ndarray, angle: int) -> np.ndarray:
    if angle % 360 == 0:
        return img_bgr
    if angle == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = img_bgr.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0,2] += (new_w/2) - center[0]
    M[1,2] += (new_h/2) - center[1]
    return cv2.warpAffine(img_bgr, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)

def mirror_image(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.flip(img_bgr, 1)

def vertical_split(img_bgr: np.ndarray, left_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    cut = int(w * left_ratio)
    left = img_bgr[:, :cut].copy()
    right = img_bgr[:, cut:].copy()
    return left, right

def horizontal_split(img_bgr: np.ndarray, top_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    cut = int(h * top_ratio)
    top = img_bgr[:cut, :].copy()
    bottom = img_bgr[cut:, :].copy()
    return top, bottom

def make_grid(img_bgr: np.ndarray, n: int) -> List[np.ndarray]:
    h, w = img_bgr.shape[:2]
    cell_w = w // n
    cell_h = h // n
    tiles = []
    for r in range(n):
        for c in range(n):
            left = c * cell_w
            top = r * cell_h
            right = (left + cell_w) if c < n-1 else w
            bottom = (top + cell_h) if r < n-1 else h
            tile = img_bgr[top:bottom, left:right].copy()
            tiles.append(tile)
    return tiles

# UI
uploaded = st.file_uploader("Upload any image (png/jpg/jpeg/webp/bmp/tiff)", type=["png","jpg","jpeg","webp","bmp","tiff"])
if not uploaded:
    st.info("Upload an image to begin. This single-file app uses OpenCV (and attempts to install it if missing).")
    st.stop()

try:
    img_bgr, filename = read_image_from_upload(uploaded)
except Exception as e:
    st.error(f"Failed to load image: {e}")
    st.stop()

orig_img = img_bgr.copy()
st.subheader("Original image (saved in variable `orig_img`)")
st.image(to_display_rgb(orig_img), use_column_width=True)

st.subheader("Image properties")
props = get_properties(orig_img, filename)
cols = st.columns(2)
with cols[0]:
    for k,v in props.items():
        st.write(f"**{k}**: {v}")

# Sidebar controls
st.sidebar.header("Operations")
do_bw = st.sidebar.checkbox("Convert to B/W (grayscale)", value=False)
rot_choice = st.sidebar.selectbox("Rotate (clockwise)", options=["0","90","180","270"], index=0)
rot_angle = int(rot_choice)
do_mirror = st.sidebar.checkbox("Mirror (horizontal flip)", value=False)
do_detect = st.sidebar.checkbox("Simple object detection (contour-based)", value=False)
min_area = st.sidebar.number_input("Min area for detection (px)", min_value=10, max_value=100000, value=500, step=10)

st.sidebar.markdown("### Splits / Cuts")
v_preset = st.sidebar.selectbox("Vertical split preset", options=["None","50-50","70-30","80-20","Custom"], index=0)
if v_preset == "Custom":
    v_percent = st.sidebar.slider("Left %", min_value=1, max_value=99, value=50)
else:
    v_map = {"None":0, "50-50":50, "70-30":70, "80-20":80}
    v_percent = v_map[v_preset]

h_preset = st.sidebar.selectbox("Horizontal split preset", options=["None","50-50","70-30","80-20","Custom"], index=0)
if h_preset == "Custom":
    h_percent = st.sidebar.slider("Top %", min_value=1, max_value=99, value=50)
else:
    h_map = {"None":0, "50-50":50, "70-30":70, "80-20":80}
    h_percent = h_map[h_preset]

grid_n = st.sidebar.number_input("Grid size n (creates n x n)", min_value=1, max_value=30, value=3, step=1)
grid_is_prime = is_prime(grid_n)
if grid_is_prime:
    st.sidebar.warning(f"{grid_n} is prime — grid disabled (per rule).")

# Apply transforms
proc = orig_img.copy()
if do_bw:
    proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
if rot_angle != 0:
    proc = rotate_image(proc, rot_angle)
if do_mirror:
    proc = mirror_image(proc)

det_boxes = []
if do_detect:
    proc_for_detect = proc if proc.ndim==3 else cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
    annotated, det_boxes = simple_object_detection(proc_for_detect, min_area=min_area)
    proc = annotated

st.subheader("Processed image")
st.image(to_display_rgb(proc), use_column_width=True)

# Download processed
try:
    proc_bytes = encode_image_to_bytes(proc, ext=".png")
    st.download_button("Download processed image", data=proc_bytes, file_name="processed.png", mime="image/png")
except Exception as e:
    st.warning(f"Failed to prepare download: {e}")

# Vertical split
st.subheader("Vertical split")
if v_percent > 0:
    left, right = vertical_split(orig_img, v_percent/100.0)
    c1, c2 = st.columns(2)
    c1.image(to_display_rgb(left), caption=f"Left — {v_percent}%")
    c2.image(to_display_rgb(right), caption=f"Right — {100-v_percent}%")
    st.download_button("Download Left", data=encode_image_to_bytes(left, ext=".png"), file_name="left.png", mime="image/png")
    st.download_button("Download Right", data=encode_image_to_bytes(right, ext=".png"), file_name="right.png", mime="image/png")
else:
    st.info("No vertical split selected.")

# Horizontal split
st.subheader("Horizontal split")
if h_percent > 0:
    top, bottom = horizontal_split(orig_img, h_percent/100.0)
    c1, c2 = st.columns(2)
    c1.image(to_display_rgb(top), caption=f"Top — {h_percent}%")
    c2.image(to_display_rgb(bottom), caption=f"Bottom — {100-h_percent}%")
    st.download_button("Download Top", data=encode_image_to_bytes(top, ext=".png"), file_name="top.png", mime="image/png")
    st.download_button("Download Bottom", data=encode_image_to_bytes(bottom, ext=".png"), file_name="bottom.png", mime="image/png")
else:
    st.info("No horizontal split selected.")

if do_detect:
    st.success(f"Detected {len(det_boxes)} object(s) (contour-based). Bounding boxes drawn on processed image.")
    if len(det_boxes) > 0:
        st.write("Boxes (x,y,w,h):")
        for b in det_boxes:
            st.write(b)

# Grid
st.subheader("Grid tiles")
if grid_is_prime:
    st.warning(f"{grid_n} is prime — grid skipped.")
else:
    if grid_n > 12:
        st.warning("Large grids may produce many tiles and slow the UI.")
    tiles = make_grid(orig_img, grid_n)
    st.write(f"Created {len(tiles)} tiles ({grid_n} x {grid_n})")
    per_row = min(4, grid_n)
    displayed = min(len(tiles), 12)
    idx = 0
    for r in range((displayed + per_row - 1) // per_row):
        cols = st.columns(per_row)
        for c in range(per_row):
            if idx < displayed:
                cols[c].image(to_display_rgb(tiles[idx]), use_column_width=True, caption=f"tile {idx}")
                idx += 1
    if len(tiles) > 0:
        st.download_button("Download tile 0", data=encode_image_to_bytes(tiles[0], ext=".png"), file_name="tile_0.png", mime="image/png")

st.info("Done. If you run into import errors on deployment (Streamlit Cloud), it usually means that runtime pip installs are blocked — use a requirements.txt with opencv-python-headless, numpy, streamlit instead.")



