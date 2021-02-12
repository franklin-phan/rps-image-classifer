import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (75,75)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('my_model.hdf5')

st.write("""
         # Rock-Paper-Scissor Hand Sign Prediction
         """
         )
description="This is a simple image classification web app to predict <span class='highlight red'>🗿&nbsp rock</span> - <span class='highlight green'>📃&nbsp paper</span> - <span class='highlight blue'>✂️ &nbsp scissor</span> hand signs"
st.markdown(description, unsafe_allow_html=True)

file = st.file_uploader('', type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        paper="It is a <span class='highlight green'>📃&nbsp paper</span> !"
        st.markdown(paper, unsafe_allow_html=True)
    elif np.argmax(prediction) == 1:
        rock="It is a <span class='highlight red'>🗿&nbsp rock</span> !"
        st.markdown(rock, unsafe_allow_html=True)
    else:
        scissor="It is a <span class='highlight blue'>✂️ &nbsp scissor</span> !"
        st.markdown(scissor, unsafe_allow_html=True)
    
    st.text("Probability (0: Paper, 1: Rock, 2: Scissor)")
    st.write(prediction)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("styles.css")