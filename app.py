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


st.markdown("<h1 id='about-us'>Poshaq: Food Recognition App</h2>", unsafe_allow_html=True)

# About Us Section
st.markdown("<h2 id='about-us'>About Us</h2>", unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    st.write(
        """
        Welcome to the Food Recognition App! Our application uses deep learning 
        to classify food items from images. Whether you are a food enthusiast 
        or just curious, our model will help you recognize various food items 
        with ease. Enjoy using the app and explore the world of food recognition!
        """
    )
with col2:
    st.image("food.jpg", caption="We recognise food!", use_container_width=True)

# Home Section
st.markdown("<h2 id='home'>Home</h2>", unsafe_allow_html=True)
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

# Contact Us Section
st.markdown("<h2 id='contact-us'>Contact Us</h2>", unsafe_allow_html=True)
st.write(
    """
    If you have any questions, feedback, or inquiries, feel free to reach out to us.
    We appreciate your support and look forward to improving our Food Recognition App!
    """
)