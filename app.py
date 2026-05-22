import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Page Setup
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

# Load Model
with st.spinner('Loading AI Model (EfficientNetB0)...'):
    model = load_model('best_model.h5')

st.title("🐱 Cat vs 🐶 Dog Classifier")
st.write("This model uses Transfer Learning (EfficientNetB0) trained with Data Augmentation.")

# Upload Widget
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess to match training
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 
    
    # Prediction
    prediction = model.predict(img_array)
    
    if prediction[0] > 0.5:
        st.success(f"Prediction: **Dog** ({prediction[0][0]*100:.2f}% confidence)")
    else:
        st.success(f"Prediction: **Cat** ({(1-prediction[0][0])*100:.2f}% confidence)")
