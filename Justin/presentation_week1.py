import streamlit as st
import speech_recognition as sr
from moviepy.editor import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import cv2
plt.style.use('ggplot')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


st.title('Motion to Emotion')



# cap= cv2.VideoCapture(0)

# width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# writer= cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))

# while True:
#     ret,frame= cap.read()

#     writer.write(frame)

#     cv2.imshow('frame', frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break
        
# cap.release()
# writer.release()
# cv2.destroyAllWindows()





# st.header('Video > Audio > Text > Sentimenet Analysis', anchor=None)

video_file = 'record.mp4'

st.video(video_file)#, format="video/mp4", start_time=0)

# audioclip = AudioFileClip(video_file)
# audioclip.write_audiofile(f"audio_from_video_13.wav")
# audio_to_be_text = (f"audio_from_video_13.wav")

# r = sr.Recognizer()

# # open the file
# with sr.AudioFile(audio_to_be_text) as source:
#     # listen for the data (load audio to memory)
#     audio_data = r.record(source)
#     # recognize (convert from speech to text)
#     text = r.recognize_google(audio_data)
    

# sentiment = sia.polarity_scores(text)

# # Compare the sentiment scores and print the most probable
# keys_to_keep = ['neg', 'neu', 'pos']
# sentiment = {k: sentiment[k] for k in keys_to_keep}
# max_emotion = (max(sentiment, key=sentiment.get))

# st.subheader('Text Extraction')
# text    

# st.subheader('Sentiment Analysis rate')
# sentiment

# st.subheader('Most Sentiment of the text')
# max_emotion


# img_file_buffer = st.camera_input("Take a picture")

# if img_file_buffer is not None:
#     # To read image file buffer with OpenCV:
#     bytes_data = img_file_buffer.getvalue()
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#     # Check the type of cv2_img:
#     # Should output: <class 'numpy.ndarray'>
#     st.write(type(cv2_img))

#     # Check the shape of cv2_img:
#     # Should output shape: (height, width, channels)
#     st.write(cv2_img.shape)



# import streamlit as st
# from streamlit_webrtc import webrtc_streamer
# import av
# import cv2


# webrtc_streamer(key="example")