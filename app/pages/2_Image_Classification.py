# app/pages/2_Image_Classification.py

import sys
from pathlib import Path

# --------------------------------------------------
# Fix PYTHONPATH so Streamlit can find `src`
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# --------------------------------------------------
# Imports
# --------------------------------------------------
import streamlit as st
from PIL import Image

from src.models.inference import run_classification

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI - Image Classification",
    layout="wide"
)

st.header("🧠 Image Classification")
st.write(
    "Upload an image containing a **single prominent object**. "
    "Predictions from all trained CNN models will be shown below."
)

# --------------------------------------------------
# File uploader
# --------------------------------------------------
uploaded = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Inference
# --------------------------------------------------
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_column_width=True
    )

    with st.spinner("🔎 Running inference on all models…"):
        results = run_classification(img)

    st.markdown("---")
    st.subheader("📊 Model Predictions")

    cols = st.columns(len(results))

    for col, (model_name, output) in zip(cols, results.items()):
        with col:
            # Model title
            st.markdown(f"### {model_name}")

            # Top-1 prediction
            st.success(
                f"**Predicted:** {output['predicted_class']}  \n"
                f"**Confidence:** {output['confidence']:.2%}"
            )

            st.markdown("**Top-5 Predictions:**")

            # Top-5 list
            for item in output["top5"]:
                st.write(
                    f"- **{item['class']}** — {item['confidence']:.2%}"
                )
