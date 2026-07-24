"""
Streamlit application for Handwritten Digit Recognition.
"""

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src.model.predictor import predict
from src.utils.image_utils import preprocess_canvas_image

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide",
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("📘 About")

st.sidebar.success("CNN trained on the MNIST dataset")

st.sidebar.markdown(
    """
### 📊 Model

- CNN (TensorFlow / Keras)

### 📂 Dataset

- MNIST

### 🎯 Accuracy

**99.34%**

---
Developed using Streamlit
"""
)

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("✍️ Handwritten Digit Recognition")

st.write(
    "Draw a handwritten digit using your mouse and click **Predict**."
)

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# ---------------------------------------------------
# Layout
# ---------------------------------------------------

left_col, right_col = st.columns([1, 1])

# ===================================================
# LEFT COLUMN
# ===================================================

with left_col:

    st.subheader("🖌️ Draw Here")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=30,
        stroke_color="white",
        background_color="black",
        width=350,
        height=350,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )

    button_col1, button_col2 = st.columns(2)

    predict_button = button_col1.button(
        "🔍 Predict",
        use_container_width=True,
    )

    clear_button = button_col2.button(
        "🧹 Clear",
        use_container_width=True,
    )

    if clear_button:
        st.session_state.canvas_key += 1
        st.rerun()

# ===================================================
# RIGHT COLUMN
# ===================================================

with right_col:

    st.subheader("Prediction")

    if predict_button:

        if canvas_result.image_data is None:
            st.warning("Please draw a digit first.")
            st.stop()

        image = canvas_result.image_data[:, :, :3].astype(np.uint8)

        processed = preprocess_canvas_image(image)

        # No digit detected
        if processed is None:
            st.warning("⚠️ Please draw a digit before predicting.")
            st.stop()

        digit, confidence, probabilities = predict(processed)

        # Metrics
        st.metric(
            label="Predicted Digit",
            value=str(digit),
        )

        st.metric(
            label="Confidence",
            value=f"{confidence * 100:.2f}%",
        )

        st.divider()

        st.subheader("📈 Probability Distribution")

        probability_df = pd.DataFrame(
            {
                "Probability": probabilities,
            },
            index=[str(i) for i in range(10)],
        )

        st.bar_chart(probability_df)

        st.divider()

        st.subheader("🖼️ Processed 28×28 Image")

        st.write("Processed Image")

        st.image(
            processed,
            width=250,
            clamp=True,
        )