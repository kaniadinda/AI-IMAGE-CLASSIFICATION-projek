import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2 as cv

IMG_SIZE = 48
MODEL_PATH = "ai_image_classifier.h5" 

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please place the trained .h5 model in this directory.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

st.title("AI Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and st.button("Check"):
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)
    img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]

    label = "FAKE (AI-generated)" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob

    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2%}")

    if label.startswith("REAL"):
        st.success("🟢 Likely REAL image")
    else:
        st.error("🔴 Likely AI-generated image")