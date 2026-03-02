import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

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
st.title("Sign Language Letter Predictor")
st.write("Upload a hand sign image to predict the letter")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

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
    st.image(image, caption="Uploaded Image")

    processed_img = preprocess_image(image)

    preds = model.predict(processed_img)
    prediction = np.argmax(preds)
    confidence = np.max(preds)

    st.success(f"Predicted Letter: {label_map.get(prediction,'Unknown')}")
    st.info(f"Confidence: {confidence*100:.2f}%")