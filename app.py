import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Load trained model (Caching keeps it fast and avoids reloading)
@st.cache_resource
def load_pneumonia_model():
    return load_model("pneumonia_model.h5")

model = load_pneumonia_model()

# 2. Preprocess function using Pillow (Cloud Safe Alternative to cv2)
def preprocess_image(uploaded_file):
    # Image ko PIL se open aur grayscale kiya (cv2.cvtColor ka alternative)
    img = Image.open(uploaded_file)
    img = ImageOps.grayscale(img)
    
    # Resize to (224, 224)
    img = img.resize((224, 224))
    
    # Normalize aur NumPy array conversion
    img_array = np.array(img) / 255.0
    
    # 3 channels me stack kiya (Jaise cv2 me kiya tha)
    img_array = np.stack((img_array,)*3, axis=-1)
    
    # Batch dimension add kiya (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# 3. User Interface
st.title("🫁 Pneumonia Detection App")
st.write("Upload an X-ray image to check for Pneumonia detection.")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # use_container_width use kiya kyunki use_column_width ab deprecate ho chuka hai
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
