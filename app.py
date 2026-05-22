import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
# IMPORT THE PREPROCESSING FUNCTION
from tensorflow.keras.applications.efficientnet import preprocess_input 

# 1. Page Setup
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

# 2. Load Model
@st.cache_resource
def load_model_cached():
    return load_model('best_model.h5')

with st.spinner('Loading AI Model (EfficientNetB0)...'):
    model = load_model_cached()

st.title("🐱 Cat vs 🐶 Dog Classifier")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # 1. Read and Reset file pointer
        uploaded_file.seek(0)
        
        # 2. Open and Convert to RGB (Fixes 4-channel issue)
        img = Image.open(uploaded_file).convert('RGB')
        
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # 3. Resize
        img = img.resize((224, 224))
        
        # 4. Convert to Array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 5. --- THE CRITICAL FIX ---
        # Use the EfficientNet specific preprocessing.
        # Do NOT manually divide by 255.0 here. 
        # This matches the math used by the ImageNet weights.
        processed_img = preprocess_input(img_array)
        
        # 6. Predict
        prediction = model.predict(processed_img)
        
        # Debug: Show the raw prediction value to verify it changes
        # st.write(f"Raw Model Output: {prediction[0][0]}")
        
        score = prediction[0][0]
        
        # Interpret Result
        if score > 0.5:
            st.success(f"Prediction: **Dog** 🐶")
            # Calculate confidence based on distance from 0.5
            conf = (score - 0.5) * 2 * 100
            st.write(f"Confidence: {conf:.2f}%")
        else:
            st.success(f"Prediction: **Cat** 🐱")
            conf = (0.5 - score) * 2 * 100
            st.write(f"Confidence: {conf:.2f}%")
            
    except Exception as e:
        st.error(f"Error: {e}")
