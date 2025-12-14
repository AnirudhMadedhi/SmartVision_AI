
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="SmartVision AI", layout="wide")

st.title("SmartVision AI")
st.write("Multi-Class Object Recognition System")

uploaded = st.file_uploader("Upload Image", type=["jpg","png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.success("Inference pipeline placeholder")
