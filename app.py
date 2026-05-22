import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. Page Config
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

# 2. Load Model
# Handles both .keras (new TF format) and .h5 (legacy format)
@st.cache_resource
def load_model_cached():
    if os.path.exists('best_model.keras'):
        return load_model('best_model.keras')
    elif os.path.exists('best_model.h5'):
        return load_model('best_model.h5')
    else:
        st.error("Model file not found! Please upload 'best_model.h5' or 'best_model.keras'.")
        st.stop()

with st.spinner('Loading AI Model (EfficientNetB0)...'):
    model = load_model_cached()

st.title("🐱 Cat vs 🐶 Dog Classifier")
st.write("Upload an image to classify it.")
st.info("ℹ️ Note: Ensure this matches your training preprocessing (Rescale 1/255).")

# 3. Upload Widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # --- FIX 1: Reset Pointer ---
        # Resets the file read position to the start so a new image can be read
        uploaded_file.seek(0)
        
        # --- FIX 2: Convert to RGB ---
        # Converts PNGs (RGBA) to RGB to match model input shape (3 channels)
        img = Image.open(uploaded_file).convert('RGB')
        
        # Show image
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # --- FIX 3: Preprocessing ---
        # 1. Resize to 224x224 (EfficientNet standard)
        img = img.resize((224, 224))
        
        # 2. Convert to numpy array
        img_array = image.img_to_array(img)
        
        # 3. Add batch dimension (Shape becomes [1, 224, 224, 3])
        img_array = np.expand_dims(img_array, axis=0)
        
        # 4. CRITICAL: Rescale pixel values
        # This MUST match your training code (rescale=1./255)
        img_array = img_array / 255.0
        
        # --- Prediction ---
        # verbose=0 stops the progress bar from appearing in the terminal
        prediction = model.predict(img_array, verbose=0)
        
        # --- FIX 4: Correct Output Extraction ---
        # Flatten the prediction array to handle different output shapes safely
        pred_score = prediction.flatten()
        
        if len(pred_score) == 2:
            # Case A: Model uses Softmax (2 outputs: [Cat_Prob, Dog_Prob])
            # We want the second value (Dog)
            score = float(pred_score[1])
        else:
            # Case B: Model uses Sigmoid (1 output: [Dog_Prob])
            # We want the first (and only) value
            score = float(pred_score[0])
        
        # --- Display Result ---
        if score >= 0.5:
            label = 'Dog'
            emoji = '🐶'
        else:
            label = 'Cat'
            emoji = '🐱'

        # Calculate confidence (Distance from 0.5)
        # e.g., 0.9 -> (0.9-0.5)*2*100 = 80% confidence
        confidence = abs(score - 0.5) * 2 * 100
        
        st.success(f"Prediction: **{label}** {emoji}")
        st.write(f"Confidence: {confidence:.2f}%")

        if confidence < 60:
            st.warning("⚠️ Low confidence: The model is unsure. Try a clearer image.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
