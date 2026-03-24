import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# -------------------------------
# 🔹 CONFIG
# -------------------------------
st.set_page_config(page_title="Malware Detection", layout="centered")

MODEL_PATH = "https://drive.google.com/file/d/1zEViAuN5VAFHV_lmOdQOYiToqoBYNwVU/view?usp=sharing"

# 🔁 Replace with your Google Drive File ID
FILE_ID = "1zEViAuN5VAFHV_lmOdQOYiToqoBYNwVU"

# -------------------------------
# 🔹 DOWNLOAD MODEL FROM DRIVE
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -------------------------------
# 🔹 CLASS NAMES (IMPORTANT)
# -------------------------------
# 🔁 Replace with your actual class names
class_names = [
    "Class1",
    "Class2",
    "Class3"
]

# -------------------------------
# 🔹 TITLE
# -------------------------------
st.title("🛡️ Proactive Malware Detection System")
st.write("Upload an image to detect malware using CNN")

# -------------------------------
# 🔹 FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# -------------------------------
# 🔹 PREDICTION FUNCTION
# -------------------------------
def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return predicted_class, confidence

# -------------------------------
# 🔹 DISPLAY RESULT
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Predicting...")

    predicted_class, confidence = predict_image(image)

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")

    # 🔥 Optional Warning
    if confidence > 0.8:
        st.error("⚠️ Malware Detected!")
    else:
        st.success("✅ Safe File")