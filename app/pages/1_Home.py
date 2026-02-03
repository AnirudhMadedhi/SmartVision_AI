# app/pages/1_Home.py
import streamlit as st

st.header("🏠 Home")

st.markdown("""
**SmartVision AI** demonstrates an end-to-end computer vision pipeline:

- Transfer learning with **VGG16, ResNet50, MobileNetV2, EfficientNetB0**
- YOLOv8 fine-tuned on custom dataset
- Real-time inference-ready architecture
""")
