import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Page Config
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

# 2. Load Model
# We use cache_resource so it only loads once
@st.cache_resource
def load_model_cached():
    return load_model('best_model.h5')

with st.spinner('Loading AI Model (EfficientNetB0)...'):
    model = load_model_cached()

st.title("🐱 Cat vs 🐶 Dog Classifier")
st.write("Upload an image to classify it.")
st.info("ℹ️ Note: This model expects RGB images scaled to 0-1 (matching training data).")

# 3. Upload Widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # --- FIX 1: Reset Pointer ---
        uploaded_file.seek(0)
        
        # --- FIX 2: Convert to RGB ---
        # Ensures PNGs with transparency don't break the model
        img = Image.open(uploaded_file).convert('RGB')
        
        # Show image
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # --- FIX 3: Preprocessing (Matches Training Code) ---
        # 1. Resize to 224x224
        img = img.resize((224, 224))
        
        # 2. Convert to array
        img_array = image.img_to_array(img)
        
        # 3. Add batch dimension (Model expects shape: [1, 224, 224, 3])
        img_array = np.expand_dims(img_array, axis=0)
        
        # 4. CRITICAL: Rescale pixel values to [0, 1]
        # This MUST match the 'rescale=1./255' used in train_datagen
        img_array = img_array / 255.0
        
        # --- Prediction ---
        prediction = model.predict(img_array)
        
        # Get the score (0 to 1)
        score = prediction[0][0]
        
        # --- Display Result ---
        # 0 = Cat, 1 = Dog
        if score > 0.5:
            st.success(f"Prediction: **Dog** 🐶")
            # Calculate confidence based on distance from 0.5
            confidence = (score - 0.5) * 2 * 100
            st.write(f"Confidence: {confidence:.2f}%")
        else:
            st.success(f"Prediction: **Cat** 🐱")
            confidence = (0.5 - score) * 2 * 100
            st.write(f"Confidence: {confidence:.2f}%")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
