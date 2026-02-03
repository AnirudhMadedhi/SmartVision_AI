# app/pages/3_Object_Detection.py

import sys
from pathlib import Path

# --------------------------------------------------
# Fix PYTHONPATH so Streamlit can find project modules
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --------------------------------------------------
# Imports
# --------------------------------------------------
import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI - Object Detection",
    layout="wide"
)

st.header("🎯 Object Detection (YOLOv8)")
st.write(
    "Upload an image and detect **multiple objects** using the trained YOLOv8 model. "
    "Adjust the confidence threshold to control detection sensitivity."
)

# --------------------------------------------------
# Load YOLO model (cached, ABSOLUTE PATH)
# --------------------------------------------------
@st.cache_resource
def load_yolo_model():
    model_path = (
        ROOT_DIR /
        "notebooks/runs/detect/runs/models/smartvision_yolo_10ep/weights/best.pt"
    )

    # Debug safety check (can remove later)
    st.write("📁 YOLO model path exists:", model_path.exists())

    return YOLO(str(model_path))

model = load_yolo_model()

# --------------------------------------------------
# Confidence threshold slider (FIXED)
# --------------------------------------------------
conf_threshold = st.slider(
    "🔧 Confidence Threshold",
    min_value=0.001,
    max_value=0.50,
    value=0.01,
    step=0.005
)

st.caption(
    "ℹ️ Lower confidence thresholds are required due to limited training epochs "
    "and CPU-based fine-tuning."
)

# --------------------------------------------------
# File uploader
# --------------------------------------------------
uploaded = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Detection
# --------------------------------------------------
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    with st.spinner("🔎 Running YOLO object detection…"):
        results = model.predict(
            source=image_np,
            conf=conf_threshold,
            save=False,
            verbose=False
        )

    result = results[0]

    # --------------------------------------------------
    # Display detected image with bounding boxes
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("📸 Detection Result")

    annotated_img = result.plot()
    st.image(
        annotated_img,
        caption="Detected Objects",
        use_column_width=True
    )

    # --------------------------------------------------
    # Display detected objects table
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Detected Objects Summary")

    if result.boxes is None or len(result.boxes) == 0:
        st.warning(
            "No objects detected at this confidence threshold. "
            "Try lowering the threshold further."
        )
    else:
        detected_data = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            detected_data.append({
                "Class": class_name,
                "Confidence": f"{conf:.2%}"
            })

        st.table(detected_data)
