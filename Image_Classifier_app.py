
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import streamlit as st


import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('keras_model.keras')

st.title("Image Content Classification")
st.write("Classify images into 'Violent', 'Adult Content', and 'Safe'.")

uploaded_file = st.file_uploader(label="Upload the image", type=['png', 'jpg', 'jpeg'], label_visibility="hidden")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    img_array = np.array(image.resize((224, 224)))/255.0
    img_array = img_array[np.newaxis, ...]  # Add batch dimension
    predictions = model.predict(img_array)
    class_names = ['Adult Content', 'Safe', 'Violent']
    st.write(f"Prediction: {class_names[np.argmax(predictions)]}")



