import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

st.set_page_config(page_title="Sign Language Letter Predictor", page_icon="✋", layout="centered")

# Custom CSS Theme
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 25px;
}

.upload-box {
    background: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result-box {
    background: #eef6ff;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: #0b6cff;
    margin-top: 15px;
}

.confidence {
    text-align: center;
    font-size: 18px;
    margin-top: 8px;
    color: #374151;
}
</style>
""", unsafe_allow_html=True)

# Load trained MLP model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("multilayer_model/multilayer_model.keras")

model = load_model()

# Label mapping (dataset skips J and Z)
label_map = {
0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',
8:'I',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',
16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',
23:'X',24:'Y'
}

# Streamlit UI
st.markdown('<div class="main-title">Sign Language Letter Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a hand sign image to predict the letter</div>', unsafe_allow_html=True)

st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# Preprocess uploaded image to match MNIST style
def preprocess_image(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,2)
    coords = cv2.findNonZero(img)
    x,y,w,h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]
    size = max(w,h)
    new_img = np.zeros((size,size), dtype=np.uint8)

    x_offset = (size - w)//2
    y_offset = (size - h)//2

    new_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
    img = new_img
    img = cv2.resize(img, (28,28))
    img = img / 255.0
    img = img.flatten()
    return img.reshape(1,-1)

# Prediction
if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_img = preprocess_image(image)

    preds = model.predict(processed_img)
    prediction = np.argmax(preds)
    confidence = np.max(preds)

    predicted_letter = label_map.get(prediction,'Unknown')

    st.markdown(
        f'<div class="result-box">Predicted Letter: {predicted_letter}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="confidence">Confidence: {confidence*100:.2f}%</div>',
        unsafe_allow_html=True
    )