import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("malware_cnn_model.h5")

# If you saved class names, load them manually
class_names = ["class1", "class2", "class3"]  # 🔁 Replace with your actual classes

# Image size
IMG_SIZE = 224

st.title("🛡️ Malware Detection System (CNN)")
st.write("Upload an image to check if it's malware")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = img_array / 255.0  # normalization
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Output
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")