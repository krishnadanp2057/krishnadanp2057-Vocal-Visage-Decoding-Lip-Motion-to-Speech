# Import all of the dependencies
import numpy as np
import streamlit as st
import os 
import imageio 

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join( 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2:
        frames, annotations = load_data(tf.convert_to_tensor(file_path))

        # Save GIF for display — don't modify original frames
        gif_frames = []
        for i in range(frames.shape[0]):
            frame = frames[i].numpy().squeeze()  # Shape: (H, W)
            frame_uint8 = np.uint8(frame * 255)
            gif_frames.append(frame_uint8)

        imageio.mimsave('animation.gif', gif_frames, fps=10)
        st.image('animation.gif', width=400)

        st.info('Original Transcription of the Video:')
        original_text = tf.strings.reduce_join(num_to_char(annotations)).numpy().decode('utf-8')
        st.text(original_text)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(frames, axis=0))  # <- Use original frames
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(tf.convert_to_tensor(decoder))).numpy().decode(
            'utf-8')
        st.text(converted_prediction)

        
