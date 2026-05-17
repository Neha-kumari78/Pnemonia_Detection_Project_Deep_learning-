
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

@st.cache_resource
def load_pneumonia_model():
    return load_model("pneumonia_model.h5")

model = load_pneumonia_model()

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = ImageOps.grayscale(img)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.stack((img_array,)*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("🫁 Pneumonia Detection App")
st.write("Upload an X-ray image to check for Pneumonia detection.")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing X-ray... Please wait..."):
        img = preprocess_image(uploaded_file)
        prediction = model.predict(img)[0][0]

        if prediction > 0.5:
            label = "NORMAL"
            confidence = prediction
        else:
            label = "PNEUMONIA"
            confidence = 1 - prediction

    st.markdown("---")
    if label == "PNEUMONIA":
        st.error(f"### Prediction: {label}")
    else:
        st.success(f"### Prediction: {label}")

    st.metric(label="Confidence Level", value=f"{confidence * 100:.2f}%")
