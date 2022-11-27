from importlib.util import spec_from_loader
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
import pickle
from pydub.silence import split_on_silence
from tqdm import tqdm
import librosa
from model import VGG
import torch, re
from torch.nn import functional as F
import math
import sys
import subprocess
import shutil
from moviepy.editor import AudioClip, VideoFileClip, concatenate_videoclips

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX



class MelTranform():
    """
    Tranformation class to get the MEL Spectrogram from the waveform
    """

    def __init__(self, n_mels=80, sample_rate=16000,
                 win_len=1024, hop_len=512, n_fft=None, to_db=True):
        """
        :param n_mels: number of MEL bands to create
        :param sample_rate: sample rate to deal with the waveforms
        :param win_len: window size of the FFT
        :param hop_len: step size of the FFT
        :param n_fft: number of FFT (default win_len)
        :param to_db: wether or not to convert to decibel
        """
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.win_len = win_len
        self.hop_len = hop_len
        self.to_db = to_db

        self.n_fft = n_fft if n_fft is not None else self.win_len

    def __call__(self, y, sample_rate=None):
        """
        Call method to convert a waveform to its MEL spectrogram representation

        :param y: waveform
        :param sample_rate: sample rate of the waveform if precised
        :return: MEL spectrogram of the waveform
        """

        # convert into float32 then change the sample rate if different from
        # the sample rate given for the class
        y = np.array(y).astype(np.float32)
        if sample_rate is not None and sample_rate != self.sample_rate:
            y = librosa.resample(y, sample_rate, self.sample_rate)

        # convert into a MEL Spectrogram
        spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            win_length=self.win_len,
            hop_length=self.hop_len,
        )

        if self.to_db:
            spec = librosa.power_to_db(spec)

        return spec

class VideoCamera():
    def __init__(self,obj,attr=0):
        self.obj= obj
        self.predper = []
        self.video = cv2.VideoCapture(attr)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        #print(self.fps)
    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        #self.predper = [np.array([0,0,0,0,0,0])]
        self.predper = [np.array([0,0,0,0,0,0,0,0])]
        facec = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__),"haarcascade_frontalface_default.xml"))
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        for (x, y, w, h) in faces:
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,225,0),2)
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (256,200),interpolation=cv2.INTER_AREA)
            image = tf.convert_to_tensor(roi[np.newaxis, :])
            image = tf.cast(image, dtype=tf.float32)/255.0
            #print(roi.size)
            self.predper = self.obj.predict_emotions(image)
        #cv2.imshow('FACIAL EMOTION DETECTION(BE IT PROJECT)',fr)
        return self.predper,fr



def gen(camera,length):
    #df = pd.DataFrame(columns=emotions)
    #S = np.zeros((length+1,8))
    S = []
    #S2 = np.zeros((length+1,8))
    S.append([camera.fps]*8)
    #S2[0] = [camera.fps]*8
    i = 1
    #print("---------------------------")
    for j in tqdm(range(1,length)):
        try:
            L,fr = camera.get_frame()
            #S2[j]=L[0]
            if np.sum(L[0]) != 0:
                S.append(L[0])
                #print(S[i])
                i+=1
        except: continue
    return np.array(S),camera.fps#,S2

def choose_vid(path_vid,obj):
    #path = os.path.dirname(__file__)+f'Videos/{nom}.mp4'
    #print(path_vid)
    cap = cv2.VideoCapture(path_vid)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print( length )
    return gen(VideoCamera(obj=obj,attr=path_vid),length)
#r = sr.Recognizer()

def process_audio(path):
    waveform, sample_rate = librosa.load(path, sr=16000)
    mel_transform = MelTranform()
    spec = mel_transform(waveform,sample_rate)
    return spec



def find_speaking(audio_clip, window_size=0.1, volume_threshold=0.4, ease_in=0.25):
    # First, iterate over audio to find all silent windows.
    num_windows = math.floor(audio_clip.end/window_size)
    window_is_silent = []
    for i in range(num_windows):
        s = audio_clip.subclip(i * window_size, (i + 1) * window_size)
        v = s.max_volume()
        window_is_silent.append(v < volume_threshold)

    # Find speaking intervals.
    speaking_start = 0
    speaking_end = 0
    speaking_intervals = []
    for i in range(1, len(window_is_silent)):
        e1 = window_is_silent[i - 1]
        e2 = window_is_silent[i]
        # silence -> speaking
        if e1 and not e2:
            speaking_start = i * window_size
        # speaking -> silence, now have a speaking interval
        if not e1 and e2:
            speaking_end = i * window_size
            new_speaking_interval = [speaking_start - ease_in, speaking_end + ease_in]
            # With tiny windows, this can sometimes overlap the previous window, so merge.
            need_to_merge = len(speaking_intervals) > 0 and speaking_intervals[-1][1] > new_speaking_interval[0]
            if need_to_merge:
                merged_interval = [speaking_intervals[-1][0], new_speaking_interval[1]]
                speaking_intervals[-1] = merged_interval
            else:
                speaking_intervals.append(new_speaking_interval)

    return speaking_intervals
