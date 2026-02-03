import streamlit as st

st.set_page_config(
    page_title="SmartVision AI",
    layout="wide"
)

st.title("SmartVision AI")
st.caption("Multi-Class Object Recognition System")

st.markdown("""
### Features
- 🧠 Image Classification (4 CNNs)
- 🎯 Object Detection (YOLOv8)
- 📊 Model Comparison Dashboard
""")

st.info("Use the sidebar to navigate ⬅️")
