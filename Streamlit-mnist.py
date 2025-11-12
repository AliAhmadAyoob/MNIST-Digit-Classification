import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv

st.set_page_config(page_title="Handwritten Digit Classifier", layout="centered")
st.title("✍️ Handwritten Digit Classification")
uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["jpg", "jpeg", "png"])
@st.cache_resource
def load_and_cache_model():
    # Model must be loaded from the saved file created in Part 1
    try:
        model = load_model('mnist_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'mnist_model.h5' is in your app directory.")
        st.stop()


network = load_and_cache_model()
if uploaded_file is not None: 
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    grayscale_img = cv.imdecode(file_bytes,0)
    grayscale_array = (np.array(grayscale_img)/255).reshape(1,784)

    network.predict(grayscale_array)
    prediction = np.argmax(network.predict(grayscale_array), axis=-1)[0]

    st.success(f"Prediction: The digit is {prediction}")
    st.balloons()