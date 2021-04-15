# importing the libraries
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.models import load_model


model = load_model("models/sleeve_length_classifier.hdf5")
classes = ['3 / 4 Sleeve', 'Short Sleeve', 'Sleeveless', 'Long Sleeve']

# Designing the interface
st.title("Automatic attribute recognition")
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
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)

# For newline
st.sidebar.write('\n')

if st.sidebar.button("Click Here to Classify"):

    if uploaded_file is None:

        st.sidebar.write("Please upload an Image to Classify")

    else:

        with st.spinner('Classifying ...'):
            # We preprocess the image to fit in algorithm.
            image = np.asarray(u_img)
            img_pixels = preprocess_input(image)
            img_pixels = cv2.resize(img_pixels, (256, 256))
            img_pixels = np.expand_dims(img_pixels, axis=0)
            preds = model.predict(img_pixels)
            prediction = np.argmax(preds)
            time.sleep(2)
            st.success('Done!')

        st.sidebar.header("Algorithm Predicts: ")

        # Formatted probability value to 3 decimal places
        probability = "{:.3f}".format(float(preds[0][prediction] * 100))

        # Classify cat being present in the picture if prediction > 0.5

        st.sidebar.write(f"Predicted sleeve length: {classes[prediction]}", '\n')
        st.sidebar.write('**Probability: **', preds[0][prediction] * 100, '%')