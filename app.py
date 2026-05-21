import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Page Setup
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

# 2. Load the Model (Cached so it doesn't reload every click)
@st.cache_resource
def load_model_cached():
    return load_model('best_model.h5')

with st.spinner('Loading AI Model (EfficientNetB0)...'):
    model = load_model_cached()

st.title("🐱 Cat vs 🐶 Dog Classifier")
st.write("Upload an image to classify it.")

# 3. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # --- CRITICAL FIX 1: Reset file pointer ---
        # This ensures we read the NEW image, not the end of the old one
        uploaded_file.seek(0)
        
        # --- CRITICAL FIX 2: Convert to RGB ---
        # Some PNGs have 4 channels (Alpha). The model needs exactly 3.
        img = Image.open(uploaded_file).convert('RGB') 
        
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Preprocessing
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0 
        
        # Debugging: Check if the input looks right (Optional)
        # st.write(f"Input shape: {img_array.shape}") 
        
        # Prediction
        prediction = model.predict(img_array)
        
        # Interpret Result
        # 0 = Cat, 1 = Dog
        score = prediction[0][0]
        
        if score > 0.5:
            st.success(f"Prediction: **Dog** 🐶")
            st.write(f"Confidence: {score * 100:.2f}%")
        else:
            st.success(f"Prediction: **Cat** 🐱")
            st.write(f"Confidence: {(1 - score) * 100:.2f}%")
            
    except Exception as e:
        st.error(f"Error processing image: {e}")
