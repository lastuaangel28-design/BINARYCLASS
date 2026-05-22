import os
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
    model_path = 'best_model.keras' if os.path.exists('best_model.keras') else 'best_model.h5'
    return load_model(model_path)

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
        prediction = model.predict(img_array, verbose=0)
        
        # Normalize prediction output into a probability score for Dog
        prediction = np.asarray(prediction).reshape(-1)
        if prediction.size == 1:
            score = float(prediction[0])
        else:
            score = float(prediction[1]) if prediction.shape[0] == 2 else float(prediction[0])
        
        # --- Display Result ---
        if score >= 0.5:
            label = 'Dog'
            emoji = '🐶'
            confidence = score * 100
        else:
            label = 'Cat'
            emoji = '🐱'
            confidence = (1.0 - score) * 100

        st.success(f"Prediction: **{label}** {emoji}")
        st.write(f"Confidence: {confidence:.2f}%")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
