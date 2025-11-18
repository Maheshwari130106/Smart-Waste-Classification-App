import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
import base64

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
MODEL_PATH = "waste_classification_model.h5"
model = load_model(MODEL_PATH)

# CLASS LABELS (must match your training generator order)
CLASS_NAMES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes',
    'trash', 'white-glass'
]

# ----------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------
def preprocess(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_disposal_instructions(label):
    instructions = {
        "battery": "Dispose in hazardous waste bin. Do NOT burn.",
        "biological": "Put in biodegradable waste or compost pit.",
        "brown-glass": "Recycle in the glass recycling bin.",
        "cardboard": "Flatten and recycle in paper/cardboard bin.",
        "clothes": "Donate if usable; else recycle at textile centers.",
        "green-glass": "Recycle in green glass bin.",
        "metal": "Recycle in metal recycling bin.",
        "paper": "Place in the paper recycling bin.",
        "plastic": "Recycle if marked; otherwise dispose as dry waste.",
        "shoes": "Donate if usable; recycle rubber parts if available.",
        "trash": "General waste — dispose in dry waste bin.",
        "white-glass": "Recycle in white glass bin."
    }
    return instructions[label]

def get_recyclability_score(label):
    scores = {
        "battery": 0.2, "biological": 0.9, "brown-glass": 0.95,
        "cardboard": 0.85, "clothes": 0.7, "green-glass": 0.95,
        "metal": 0.9, "paper": 0.8, "plastic": 0.6, 
        "shoes": 0.3, "trash": 0.1, "white-glass": 0.95
    }
    return scores[label]

def get_carbon_footprint(label):
    impact = {
        "battery": 5.0, "biological": 0.5, "brown-glass": 1.2,
        "cardboard": 0.8, "clothes": 2.5, "green-glass": 1.2,
        "metal": 3.0, "paper": 0.7, "plastic": 4.0,
        "shoes": 2.2, "trash": 4.5, "white-glass": 1.2
    }
    return impact[label]

def estimate_quantity(img):
    # fake simple estimation
    return np.random.randint(50, 500)

# ----------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------
st.title("♻️ Smart Waste Classification System")
st.write("Upload an image to classify the type of waste and get recycling insights.")

uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    img_array = preprocess(img)
    preds = model.predict(img_array)[0]

    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    recyclability = get_recyclability_score(predicted_class)
    carbon = get_carbon_footprint(predicted_class)
    quantity = estimate_quantity(img)
    disposal = get_disposal_instructions(predicted_class)

    st.success(f"### 🟩 Predicted Waste Type: **{predicted_class.upper()}**")
    st.write(f"### 🔍 Confidence: **{confidence*100:.2f}%**")
    st.write(f"### ♻️ Recyclability Score: **{recyclability*100:.1f}%**")
    st.write(f"### 🌍 Carbon Footprint Impact: **{carbon} kg CO₂**")
    st.write(f"### 📦 Estimated Waste Quantity: **{quantity} grams**")
    st.info(f"### 📝 Disposal Instructions:\n{disposal}")

    st.write("---")
    st.write("### 📊 Prediction Probabilities")
    st.bar_chart(preds)

