# importing the libraries
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import psutil
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.models import load_model
import gc
import keras

@st.cache(allow_output_mutation=True)
def init_sl_model():
    model = load_model("models/sleeve_length_classifier.hdf5")
    return model

def predict_sleeve_length(img_pixels_tensor):
    model = init_sl_model()
    preds = model.predict(img_pixels_tensor)

    return preds

st.sidebar.write(psutil.virtual_memory())

classes = ['3 / 4 Sleeve', 'Short Sleeve', 'Sleeveless', 'Long Sleeve']
# Designing the interface
# For newline
st.write('\n')

image = Image.open('images/image.png')
show = st.image(image, use_column_width=True)


st.sidebar.title("Upload Image")

# Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
# Choose your own image
uploaded_file = st.sidebar.file_uploader(" ", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    with st.spinner('Loading...'):
        u_img = Image.open(uploaded_file)
        show.image(u_img, 'Uploaded Image', use_column_width=True)

# For newline
st.sidebar.write('\n')

if st.sidebar.button("Click Here to Classify"):

    if uploaded_file is None:

        st.sidebar.write("Please upload an Image to Classify")

    else:

        with st.spinner('Classifying ...'):
            try:
                # We preprocess the image to fit in algorithm.
                image = np.asarray(u_img)
                img_pixels = preprocess_input(image)
                img_pixels = cv2.resize(img_pixels, (256, 256))
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels_tensor = tf.convert_to_tensor(img_pixels, dtype=tf.int32)
                preds = predict_sleeve_length(img_pixels_tensor)
                prediction = np.argmax(preds)
                st.success('Done!')
                st.sidebar.header("Algorithm Predicts: ")
                probability = "{:.3f}".format(float(preds[0][prediction] * 100))
                st.sidebar.write(f"Predicted sleeve length: {classes[prediction]}", '\n')
                st.sidebar.write('**Probability: **', preds[0][prediction] * 100, '%')
                # gc.collect()
                # keras.backend.clear_session()
            except Exception as e:
                st.sidebar.error(e)
                st.sidebar.error("This file format is not supported. Please try to upload another image...")

