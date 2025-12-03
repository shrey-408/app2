# app.py
import io
from PIL import Image, ImageOps
import streamlit as st

st.set_page_config(page_title="Simple Photo Uploader", page_icon="🖼️", layout="centered")

st.title("Simple Photo Uploader — upload any photo (yes, any)")
st.write("Upload an image, preview it, convert to grayscale, resize, and download the result.")

uploaded = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"])

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGBA")  # keep alpha if present
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    st.subheader("Original image")
    st.image(image, use_column_width=True)

    st.sidebar.header("Options")
    do_gray = st.sidebar.checkbox("Convert to grayscale", value=False)
    keep_aspect = st.sidebar.checkbox("Keep aspect ratio when resizing", value=True)
    width = st.sidebar.number_input("Resize width (px, 0 = keep original)", min_value=0, value=0, step=10)
    height = st.sidebar.number_input("Resize height (px, 0 = keep original)", min_value=0, value=0, step=10)

    # Process image copy
    img = image.convert("RGBA")

    if do_gray:
        # Convert to grayscale but keep alpha if present
        # Remove alpha for processing then paste alpha back
        if img.mode == "RGBA":
            alpha = img.split()[-1]
            rgb = img.convert("RGB")
            gray = ImageOps.grayscale(rgb)
            img = Image.merge("RGBA", (gray, gray, gray, alpha))
        else:
            img = ImageOps.grayscale(img).convert("RGBA")

    if width > 0 or height > 0:
        orig_w, orig_h = img.size
        if keep_aspect:
            # determine new size keeping aspect ratio
            if width > 0 and height > 0:
                # choose scale that fits within given box
                scale = min(width / orig_w, height / orig_h)
                new_w = max(1, int(orig_w * scale))
                new_h = max(1, int(orig_h * scale))
            elif width > 0:
                scale = width / orig_w
                new_w = width
                new_h = max(1, int(orig_h * scale))
            elif height > 0:
                scale = height / orig_h
                new_h = height
                new_w = max(1, int(orig_w * scale))
        else:
            new_w = width if width > 0 else orig_w
            new_h = height if height > 0 else orig_h

        img = img.resize((new_w, new_h), Image.LANCZOS)

    st.subheader("Processed image")
    st.image(img, use_column_width=True)

    # Prepare file for download (preserve original format if possible)
    def get_download_bytes(pil_img, fmt=None):
        buf = io.BytesIO()
        # prefer original format; default to PNG
        fmt = fmt or "PNG"
        # If format is JPEG and image has alpha, convert
        if fmt.upper() in ("JPG", "JPEG") and pil_img.mode == "RGBA":
            background = Image.new("RGB", pil_img.size, (255, 255, 255))
            background.paste(pil_img, mask=pil_img.split()[3])  # 3 is alpha
            background.save(buf, format="JPEG", quality=95)
        else:
            save_img = pil_img
            # remove alpha for formats that don't support it
            if fmt.upper() in ("JPG", "JPEG", "BMP") and pil_img.mode == "RGBA":
                save_img = pil_img.convert("RGB")
            save_img.save(buf, format=fmt)
        buf.seek(0)
        return buf

    # Guess original format from uploaded filename
    orig_fmt = None
    if uploaded.name.lower().endswith(".jpg") or uploaded.name.lower().endswith(".jpeg"):
        orig_fmt = "JPEG"
    elif uploaded.name.lower().endswith(".png"):
        orig_fmt = "PNG"
    elif uploaded.name.lower().endswith(".webp"):
        orig_fmt = "WEBP"
    elif uploaded.name.lower().endswith(".bmp"):
        orig_fmt = "BMP"
    elif uploaded.name.lower().endswith(".tiff") or uploaded.name.lower().endswith(".tif"):
        orig_fmt = "TIFF"

    fmt = st.selectbox("Download format", options=[f for f in ["PNG", "JPEG", "WEBP", "BMP", "TIFF"]], index=0)

    download_buf = get_download_bytes(img, fmt=fmt)
    suggested_name = f"processed_image.{fmt.lower()}"
    st.download_button("Download processed image", data=download_buf, file_name=suggested_name, mime=f"image/{fmt.lower()}")
else:
    st.info("Upload an image to get started. Any photo — doesn't matter how chaotic it is.")
