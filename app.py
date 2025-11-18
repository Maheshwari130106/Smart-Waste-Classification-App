import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("waste_classification_model.h5")

# Class names
classes = [
    'battery',
    'biological',
    'brown-glass',
    'cardboard',
    'clothes',
    'green-glass',
    'metal',
    'paper',
    'plastic',
    'shoes',
    'trash',
    'white-glass'
]

st.title("♻️ Smart Waste Classification App")
st.write("Upload an image and the model will classify the waste type.")

uploaded_file = st.file_uploader("Choose a waste image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    st.success(f"### 🟢 Predicted Waste Type: **{predicted_class.upper()}**")
