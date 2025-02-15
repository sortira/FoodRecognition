import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((180, 180))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  #  Normalize
    return img_array

# Streamlit UI
st.title("Food Recognition App")
st.write("Upload an image of food and the model will predict what it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    st.write("Classifying...")
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    
    predicted_class = np.argmax(prediction)
    class_names = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
    result = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"
    st.write(f"Prediction: **Class {result}**")
