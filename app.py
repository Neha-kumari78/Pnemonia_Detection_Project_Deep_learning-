import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model("pneumonia_model.h5")

# Preprocess function
def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    
    img = np.stack((img,)*3, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    return img

# UI
st.title("🫁 Pneumonia Detection App")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    img = preprocess_image(uploaded_file)
    prediction = model.predict(img)[0][0]
    
    if prediction > 0.5:
        label = "NORMAL"
        confidence = prediction
    else:
        label = "PNEUMONIA"
        confidence = 1 - prediction
    
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")