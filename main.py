# importing the libraries
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.models import load_model
import operator
from colormap import rgb2hex
import pickle


def get_dress(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = tf.image.resize_with_pad(img, target_height=512, target_width=512)
    bgr = img.numpy()
    img = np.expand_dims(img, axis=0) / 255.
    with tf.device('/gpu'):
        seq = model.predict(img)
    seq_new = seq[3][0, :, :, 0]
    mask = np.where(seq_new > 0.02, 1, 0)
    pixels = bgr[:, :, ::-1][mask != 0]

    return pixels


def calculate_colors(pixels, main_color_names, main_colors_cashe):
    count = 0
    main_stat = {}
    for name in main_color_names:
        main_stat[name] = 0

    for i, pixel in enumerate(pixels):
        pixel_hex = rgb2hex(int(pixel[0]), int(pixel[1]), int(pixel[2]))
        main_color_name = main_colors_cashe[pixel_hex]
        main_stat[main_color_name] += 1
        count += 1
    for k, v in main_stat.items():
        main_stat[k] = round(v / count, 2)

    main_stat = sorted(main_stat.items(), key=operator.itemgetter(0))

    return main_stat


@st.cache(allow_output_mutation=True)
def load_color_classifier():
    with open('models/colors_cache.pickle', 'rb') as f:
        cache = pickle.load(f)

    with open('models/color_names.pickle', 'rb') as f:
        names = pickle.load(f)

    return cache, names



@st.cache(allow_output_mutation=True)
def load_segmentation_model():
    with tf.device('/gpu'):
        model = load_model("models/segmentation_model.h5")
    return model

@st.cache(allow_output_mutation=True)
def load_sleeve_length_classifier():
    with tf.device('/gpu'):
        model = load_model("models/sleeve_length_classifier.hdf5")
    return model

@st.cache(allow_output_mutation=True)
def load_dress_length_classifier():
    with tf.device('/gpu'):
        model = load_model("models/dress_length_classifier.hdf5")
    return model

def predict_sleeve_length(img_pixels_tensor, model):
    with tf.device('/gpu'):
        preds = model.predict(img_pixels_tensor)

    return preds

def predict_dress_length(img_pixels_tensor, model):
    with tf.device('/gpu'):
        preds = model.predict(img_pixels_tensor)

    return preds


def parse_colors(colors):
    max = -np.inf
    for i, color in enumerate(colors):
        if color[1] > max:
            max = color[1]
            color_prediction = color
    return color_prediction


try:
    cache, names = load_color_classifier()
    sl_model = load_sleeve_length_classifier()
    dl_model = load_dress_length_classifier()
    seg_model = load_segmentation_model()
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
                    pixels = get_dress(image, seg_model)
                    result = calculate_colors(pixels, names, cache)
                    color_name, color_score = parse_colors(result)
                    img_pixels = preprocess_input(image)
                    img_pixels = cv2.resize(img_pixels, (256, 256))
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels_tensor = tf.convert_to_tensor(img_pixels, dtype=tf.int32)
                    sl_preds = predict_sleeve_length(img_pixels_tensor, sl_model)
                    dl_preds = predict_dress_length(img_pixels_tensor, dl_model)
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
                    st.sidebar.write('\n')
                    st.sidebar.write(f'Predicted dress color: **{color_name}**')
                    st.sidebar.write('\n')
                    st.sidebar.write('Probability: ', color_score * 100, '%')
                except Exception as e:
                    st.sidebar.error(e)
                    st.sidebar.error("This file format is not supported. Please try to upload another image...")
except Exception as e:
    print(e)
    print('Restarting server...')
    st.experimental_rerun()

