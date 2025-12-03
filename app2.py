# app.py
import io
import math
from typing import Tuple, List

import numpy as np
import cv2
from PIL import Image, ImageOps
import streamlit as st

st.set_page_config(page_title="Image Toolbox", page_icon="🧰", layout="wide")
st.title("Image Toolbox — classical CV edition (no DL)")

# -----------------------
# Utilities
# -----------------------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    if img.mode == "RGBA":
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        return Image.fromarray(img)
    # BGR / BGRA -> RGB / RGBA
    if img.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    r = int(math.sqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True

def get_image_properties(pil_img: Image.Image, cv_img: np.ndarray) -> dict:
    props = {}
    props["Format"] = pil_img.format or "Unknown"
    props["Mode"] = pil_img.mode
    props["Size (WxH)"] = f"{pil_img.width} x {pil_img.height}"
    props["Channels (cv)"] = cv_img.shape[2] if cv_img.ndim == 3 else 1
    props["Dtype (cv)"] = str(cv_img.dtype)
    props["Num pixels"] = pil_img.width * pil_img.height
    return props

def simple_object_detection(cv_img: np.ndarray, min_area=500) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
    """
    Simple object detection using contours:
    - Convert to gray
    - Blur
    - Canny or adaptive thresh
    - findContours
    Returns annotated image (BGR) and list of bboxes (x,y,w,h)
    """
    if cv_img.ndim == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img.copy()

    # Blur and detect edges
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Use adaptive threshold so different lighting works
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

    # Morphology to join pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    out = cv_img.copy() if cv_img.ndim == 3 else cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        boxes.append((x,y,w,h))
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(out, f"{int(area)}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return out, boxes

def crop_vertical(pil_img: Image.Image, ratio: float) -> Tuple[Image.Image, Image.Image]:
    w, h = pil_img.size
    cut = int(w * ratio)
    left = pil_img.crop((0,0,cut,h))
    right = pil_img.crop((cut,0,w,h))
    return left, right

def crop_horizontal(pil_img: Image.Image, ratio: float) -> Tuple[Image.Image, Image.Image]:
    w, h = pil_img.size
    cut = int(h * ratio)
    top = pil_img.crop((0,0,w,cut))
    bottom = pil_img.crop((0,cut,w,h))
    return top, bottom

def make_grid(pil_img: Image.Image, grid_n: int) -> List[Image.Image]:
    w, h = pil_img.size
    cell_w = w // grid_n
    cell_h = h // grid_n
    tiles = []
    for r in range(grid_n):
        for c in range(grid_n):
            left = c * cell_w
            upper = r * cell_h
            right = left + cell_w if (c < grid_n-1) else w
            lower = upper + cell_h if (r < grid_n-1) else h
            tile = pil_img.crop((left, upper, right, lower))
            tiles.append(tile)
    return tiles

# -----------------------
# UI: Upload
# -----------------------
uploaded = st.file_uploader("Upload any image (png/jpg/jpeg/webp/bmp/tiff)", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"])

if not uploaded:
    st.info("Upload an image to begin. The app will let you view, transform, crop, and run simple object detection (no DL).")
    st.stop()

# Read image (step 1)
pil_img = Image.open(uploaded).convert("RGBA") if uploaded.name.lower().endswith("png") else Image.open(uploaded).convert("RGB")
cv_img = pil_to_cv(pil_img)

# Show original image saved in variable (step 2)
st.subheader("Original image")
col1, col2 = st.columns([1,1])
with col1:
    st.image(pil_img, use_column_width=True)
with col2:
    st.markdown("### Properties")
    properties = get_image_properties(pil_img, cv_img)
    for k,v in properties.items():
        st.write(f"**{k}**: {v}")

# Controls panel
st.sidebar.header("Operations (pick any)")
# 3 Change color to bw
do_bw = st.sidebar.checkbox("Convert to B/W (grayscale)")

# 5 Rotate
st.sidebar.markdown("**Rotate**")
rot = st.sidebar.selectbox("Rotate by", options=["0","90","180","270"], index=0)

# 6 Mirror
do_mirror = st.sidebar.checkbox("Mirror (horizontal flip)")

# 7 Object detection (classical CV)
do_detect = st.sidebar.checkbox("Simple object detection (contour-based)")

# 8/9/10 Cuts
st.sidebar.markdown("**Cuts / Splits**")
vcut = st.sidebar.selectbox("Vertical split preset", options=["None","50-50","70-30","80-20","Custom"])
if vcut == "Custom":
    v_ratio = st.sidebar.slider("Left portion (%)", min_value=1, max_value=99, value=50)
else:
    v_ratio = {"None":0, "50-50":50, "70-30":70, "80-20":80}[vcut]

hcut = st.sidebar.selectbox("Horizontal split preset", options=["None","50-50","70-30","80-20","Custom"])
if hcut == "Custom":
    h_ratio = st.sidebar.slider("Top portion (%)", min_value=1, max_value=99, value=50)
else:
    h_ratio = {"None":0, "50-50":50, "70-30":70, "80-20":80}[hcut]

# 11 Grid
st.sidebar.markdown("**Grid**")
grid_n = st.sidebar.number_input("Grid size (n — creates n x n tiles)", min_value=1, max_value=50, value=3, step=1)
# warn if prime
if is_prime(grid_n):
    st.sidebar.warning(f"{grid_n} is prime — grid disabled (you asked for no grids for primes).")

# Apply transformations in order
proc = pil_img.copy()

# Step 3: grayscale
if do_bw:
    proc = proc.convert("L").convert("RGB")  # keep 3 channels for consistency

# Step 5: rotate
if rot != "0":
    angle = int(rot)
    # PIL rotate uses counter-clockwise; use expand to keep full image
    proc = proc.rotate(-angle, expand=True)

# Step 6: mirror
if do_mirror:
    proc = ImageOps.mirror(proc)

# Convert to cv for further ops (object detection)
proc_cv = pil_to_cv(proc)

# Step 7: object detection
detected_boxes = []
if do_detect:
    annotated_cv, boxes = simple_object_detection(proc_cv, min_area=500)
    proc = cv_to_pil(annotated_cv)
    detected_boxes = boxes

# Show processed image and allow download
st.subheader("Processed image")
st.image(proc, use_column_width=True)

# Download button
buf = io.BytesIO()
proc_format = "PNG"
proc.save(buf, format=proc_format)
buf.seek(0)
st.download_button("Download processed image", data=buf, file_name=f"processed.{proc_format.lower()}", mime=f"image/{proc_format.lower()}")

# Step 8: Cut image vertically 50-50 (or based on selection)
st.subheader("Vertical split")
if v_ratio > 0:
    left, right = crop_vertical(pil_img, v_ratio/100.0)
    c1, c2 = st.columns(2)
    c1.image(left, caption=f"Left — {v_ratio}%")
    c2.image(right, caption=f"Right — {100-v_ratio}%")
    # download individually
    b1 = io.BytesIO(); left.save(b1, format="PNG"); b1.seek(0)
    b2 = io.BytesIO(); right.save(b2, format="PNG"); b2.seek(0)
    st.download_button("Download Left part", data=b1, file_name="left.png")
    st.download_button("Download Right part", data=b2, file_name="right.png")
else:
    st.info("No vertical split selected (choose a preset or custom).")

# Step 9: Horizontal split
st.subheader("Horizontal split")
if h_ratio > 0:
    top, bottom = crop_horizontal(pil_img, h_ratio/100.0)
    c1, c2 = st.columns(2)
    c1.image(top, caption=f"Top — {h_ratio}%")
    c2.image(bottom, caption=f"Bottom — {100-h_ratio}%")
    b1 = io.BytesIO(); top.save(b1, format="PNG"); b1.seek(0)
    b2 = io.BytesIO(); bottom.save(b2, format="PNG"); b2.seek(0)
    st.download_button("Download Top part", data=b1, file_name="top.png")
    st.download_button("Download Bottom part", data=b2, file_name="bottom.png")
else:
    st.info("No horizontal split selected (choose a preset or custom).")

# Step 11: Grid
st.subheader("Grid tiles")
if is_prime(grid_n):
    st.warning(f"{grid_n} is prime — grid creation skipped (you requested that).")
else:
    if grid_n > 20:
        st.warning("Large grids may produce many tiles — performance may slow down.")
    tiles = make_grid(pil_img, grid_n)
    st.write(f"Created {len(tiles)} tiles ({grid_n} x {grid_n})")
    # Display tiles in rows using st.columns
    per_row = min(6, grid_n)  # display up to 6 per row for readability
    idx = 0
    for r in range(grid_n):
        cols = st.columns(per_row)
        for c in range(per_row):
            if idx < len(tiles):
                cols[c].image(tiles[idx].resize((int(pil_img.width//grid_n), int(pil_img.height//grid_n))), use_column_width=True)
                idx += 1
    # Offer a zip download? For simplicity provide first tile download
    b = io.BytesIO()
    tiles[0].save(b, format="PNG")
    b.seek(0)
    st.download_button("Download first tile (example)", data=b, file_name="tile_0.png")

# Final summary of object detection if performed
if do_detect:
    st.success(f"Detected {len(detected_boxes)} object(s) (contour-based). Bounding boxes shown on processed image.")
    if len(detected_boxes) > 0:
        st.write("Boxes (x, y, w, h):")
        for b in detected_boxes:
            st.write(b)

