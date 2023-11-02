import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL_PATH = './model'

model = tf.keras.models.load_model(MODEL_PATH) 

# Define the class labels
class_labels = ['Cat', 'Dog']

# Create a Streamlit app
st.title('Cat or Dog Classifier')
st.write('Upload an image to check if it\'s a cat or a dog.')

uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    image = image.resize((28, 28))
    image = np.asarray(image)
    image = image.reshape(1, 28, 28, 1)
    
    # Make predictions
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]

    st.write(f'Prediction: {predicted_class}')
