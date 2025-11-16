import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your model
model = load_model("waste_classification_model.h5")

# Class labels
classes = [
    'metal', 'white-glass', 'brown-glass', 'paper', 'trash',
    'cardboard', 'clothes', 'biological', 'shoes', 'plastic',
    'battery', 'green-glass'
]

st.title("♻️ Smart Waste Classification App")
st.write("Upload an image and find out the waste category!")

uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

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
