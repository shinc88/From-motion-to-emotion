import streamlit as st 
from pyngrok import ngrok
from google.colab import drive
from google.colab import files
from google.colab import output
import pandas as pd
import speech_recognition as sr
import base64
from base64 import b64decode
import cv2
from deepface import DeepFace 
from collections import Counter
from pathlib import Path
import pickle
import os
import glob
from collections import Counter
from PIL import Image
import moviepy.editor as mp
r = sr.Recognizer()
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
import numpy as np
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
import librosa
import librosa.display
import matplotlib.pyplot as plt
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

st.title('Motion to Emotion')

image = Image.open('/content/gdrive/MyDrive/From Motion to Emotion/emotion_imgage.jpeg')
st.image(image)#, caption='Sunrise by the mountains')




import uuid
from pathlib import Path

import av
import cv2
import streamlit as st
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # perform edge detection
    img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


RECORD_DIR = Path("./records")
RECORD_DIR.mkdir(exist_ok=True)


def app():
    if "prefix" not in st.session_state:
        st.session_state["prefix"] = str(uuid.uuid4())
    prefix = st.session_state["prefix"]
    in_file = RECORD_DIR / f"{prefix}_input.flv"
    out_file = RECORD_DIR / f"{prefix}_output.flv"

    def in_recorder_factory() -> MediaRecorder:
        return MediaRecorder(
            str(in_file), format="flv"
        )  # HLS does not work. See https://github.com/aiortc/aiortc/issues/331

    def out_recorder_factory() -> MediaRecorder:
        return MediaRecorder(str(out_file), format="flv")

    webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": True,
            "audio": True,
        },
        video_frame_callback=video_frame_callback,
        in_recorder_factory=in_recorder_factory,
        out_recorder_factory=out_recorder_factory,
    )

    if in_file.exists():
        with in_file.open("rb") as f:
            st.download_button(
                "Download the recorded video without video filter", f, "input.flv"
            )
    if out_file.exists():
        with out_file.open("rb") as f:
            st.download_button(
                "Download the recorded video with video filter", f, "output.flv"
            )


if __name__ == "__main__":
    app()






option = st.selectbox(
    'Select a video to analyse',
    ('Choose video',
    'Greta Thunberg\'s speech at UN', 
    'Emily Esfahani Smith - There\'s more to life than being happy', 
    'Matthew McConaughey winning Best Actor 86th Oscars (2014)'))


if option == 'Greta Thunberg\'s speech at UN':
  path = '/content/gdrive/MyDrive/From Motion to Emotion/Greta_UN.mp4'
  #st.subheader('Greta Thunberg\'s speech at UN')
  st.video(path)

if option == 'Emily Esfahani Smith - There\'s more to life than being happy':
  path = '/content/gdrive/MyDrive/From Motion to Emotion/Emily_Happy.mp4'
  #st.subheader('Emily Esfahani Smith - There\'s more to life than being happy')
  st.video(path)

if option == 'Matthew McConaughey winning Best Actor 86th Oscars (2014)':
  path = '/content/gdrive/MyDrive/From Motion to Emotion/Matthew_Oscars.mp4'
  #st.subheader('Matthew McConaughey winning Best Actor 86th Oscars (2014)')
  st.video(path)



if st.button('Start Analysis'):

# Speech Recognition 
    
  clip = mp.VideoFileClip(path) 
  clip.audio.write_audiofile('audio.wav')

  audio = sr.AudioFile("audio.wav")

  with audio as source:
    audio_file = r.record(source)
    text = r.recognize_google(audio_file)


# Face emotion dectection 
# Extrac frames from video 
  video = cv2.VideoCapture(path)
      
  total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

  step = 30 # if step = 10, then only every 10-th frame will be considered and saved

  from collections import Counter
  count = 0

  frame_list = [] 

  for i in range(total_frames):
      ret, frame = video.read()
      if ret:
          count += 1
          if count % step == 0:
              frame_list.append(frame)
              
  video.release()

# Analysis emotion from frames

  df = pd.DataFrame()
  emotions_list = []

  for picture in frame_list:
    objs = DeepFace.analyze(picture, actions = 'emotion', enforce_detection = False)
    dominant_emotion = objs[0]['dominant_emotion']
    emotions_list.append(dominant_emotion)

# Create datafram from the output

  df['Emotion'] = Counter(emotions_list).keys()
  df['Counts'] = Counter(emotions_list).values()
  df['Percentage'] = (df['Counts']/df['Counts'].sum())*100
    
  df_sorted = df[['Emotion','Percentage']].sort_values(by = 'Percentage', ascending=False)
  df_sorted.reset_index(drop = True, inplace=True)

# Dictionary

  Face_emo = {'angry' : 'Anger',
              'fear' : 'Fear',
              'neutral' : 'Neutral', 
              'sad' : 'Sad', 
              'disgust' : 'Disgust', 
              'happy' : 'Happy', 
              'surprise' : 'Suprise'
              }

  emoji_dic = {'neutral': ':neutral_face:',
            'fear':':anguished:',
            'angry': ':angry:',
            'sad':':cry:',
            'happy':':grin:',
            'suprise':':astanished:',
            'disgust':':confounded:',
            'positive' : ':grinning:',
            'negative' : ':worried:'
            }

  
  text_sent = {'neg' : 'Negative',
              'neu' : 'Neutral',
              'pos' : 'Positive' 
              }
  
  text_emo = {'joy' : 'Happy',
              'sadness' : 'Sad',
              'fear' : 'Fear',
              'anger' : 'Anger',
              'suprise' : 'Surprise',
              'neutral' : 'Neutral',
              'disgust' : 'Disgust'
              }

  
# Result displaying 

  with st.container():
    st.header(':blue[Speech Recognition]')
    st.write(text)

# Text sentiment analysis 
    sentiment = sia.polarity_scores(text)
    keys_to_keep = ['neg', 'neu', 'pos']
    sentiment = {k: sentiment[k] for k in keys_to_keep}
    st.header(':blue[Sentiment of the Text]')
    
    #st.subheader(text_sent[max(sentiment, key=sentiment.get)])
    col1, col2, col3 = st.columns(3)
    with col1:
      st.write('Negative')
      st.subheader(sentiment['neg'])
      st.subheader(emoji_dic['negative'])
    with col2:
     st.write('Neutral')
     st.subheader(sentiment['neu'])
     st.subheader(emoji_dic['neutral'])
    with col3:
     st.write('Positive')
     st.subheader(sentiment['pos'])
     st.subheader(emoji_dic['positive'])

# Text Emotion analysis

    st.header(':blue[Emotion of the Text]')
    loaded_model = pickle.load(open('emo_text_model.sav', 'rb'))
    emo_predict = loaded_model.predict([text])[0]
    st.subheader(text_emo[emo_predict])

  with st.container():
    st.header(":blue[Face Emotion]")
    col1, col2, col3 = st.columns(3)
    
    with col1:
      st.subheader('Most domiant')
      st.subheader(Face_emo[df_sorted.Emotion[0]])
      st.subheader(emoji_dic[df_sorted.Emotion[0]])
      st.subheader(df_sorted.Percentage[0].round(2))

    with col2:
      st.subheader('2nd')
      st.subheader(Face_emo[df_sorted.Emotion[1]])
      st.subheader(emoji_dic[df_sorted.Emotion[1]])
      st.subheader(df_sorted.Percentage[1].round(2))

    with col3:
      st.subheader('3rd')
      st.subheader(Face_emo[df_sorted.Emotion[2]])
      st.subheader(emoji_dic[df_sorted.Emotion[2]])
      st.subheader(df_sorted.Percentage[2].round(2))

    with st.container():
      st.header(':blue[Spectrogram of the voice]')
      y, sr = librosa.load('audio.wav')
      y, index = librosa.effects.trim(y)  
      mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=100)
      mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
      spectrogram = librosa.display.specshow(mel_spect, y_axis='mel', fmax=20000, x_axis='time');
      plt.savefig('spectrogram.png')
      st.image('spectrogram.png')