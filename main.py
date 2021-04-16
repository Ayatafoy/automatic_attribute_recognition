# importing the libraries
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.models import load_model

@st.cache(allow_output_mutation=True)
def init_sl_model():
    model = load_model("models/sleeve_length_classifier.hdf5")
    return model

@st.cache(allow_output_mutation=True)
def init_dl_model():
    model = load_model("models/dress_length_classifier.hdf5")
    return model

def predict_sleeve_length(img_pixels_tensor):
    with tf.device('/cpu'):
        model = init_sl_model()
        preds = model.predict(img_pixels_tensor)

    return preds

def predict_dress_length(img_pixels_tensor):
    with tf.device('/cpu'):
        model = init_dl_model()
        preds = model.predict(img_pixels_tensor)

    return preds

try:
    sl_classes = ['3 / 4 Sleeve', 'Short Sleeve', 'Sleeveless', 'Long Sleeve']
    dl_classes = ['long', 'short', 'midi', 'knee']
    st.write('\n')

    image = Image.open('images/image.png')
    show = st.image(image, use_column_width=True)
    st.sidebar.title("Upload Image")

    st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_file = st.sidebar.file_uploader(" ", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        with st.spinner('Loading...'):
            u_img = Image.open(uploaded_file)
            show.image(u_img, 'Uploaded Image', use_column_width=True)

    st.sidebar.write('\n')

    if st.sidebar.button("Click Here to Classify"):

        if uploaded_file is None:

            st.sidebar.write("Please upload an Image to Classify")

        else:
            with st.spinner('Classifying ...'):
                try:
                    image = np.asarray(u_img)
                    if image.shape[2] == 4:
                        image = image[:, :, :3]
                    img_pixels = preprocess_input(image)
                    img_pixels = cv2.resize(img_pixels, (256, 256))
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels_tensor = tf.convert_to_tensor(img_pixels, dtype=tf.int32)
                    sl_preds = predict_sleeve_length(img_pixels_tensor)
                    dl_preds = predict_dress_length(img_pixels_tensor)
                    sl_prediction = np.argmax(sl_preds)
                    dl_prediction = np.argmax(dl_preds)
                    st.success('Done!')
                    st.sidebar.header("Algorithm Predicts: ")
                    sl_probability = "{:.3f}".format(float(sl_preds[0][sl_prediction] * 100))
                    dl_probability = "{:.3f}".format(float(dl_preds[0][dl_prediction] * 100))
                    st.sidebar.write(f'Predicted sleeve length: **{sl_classes[sl_prediction]}**')
                    st.sidebar.write('\n')
                    st.sidebar.write('Probability: ', sl_preds[0][sl_prediction] * 100, '%')
                    st.sidebar.write('\n')
                    st.sidebar.write(f'Predicted dress length: **{dl_classes[dl_prediction]}**')
                    st.sidebar.write('\n')
                    st.sidebar.write('Probability: ', dl_preds[0][dl_prediction] * 100, '%')
                except Exception as e:
                    st.sidebar.error(e)
                    st.sidebar.error("This file format is not supported. Please try to upload another image...")
except Exception as e:
    print(e)
    print('Restarting server...')
    st.experimental_rerun()

