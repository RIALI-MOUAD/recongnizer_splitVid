import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import gc


r = sr.Recognizer()

def get_small_audio(path,language="en-EN"):
    print(path)
    r = sr.Recognizer()
    #print('Recognizer set!!')
    with sr.AudioFile(path) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        #print(audio_data)
        # recognize (convert from speech to text)
        #print('Audio Recorded!!')
        text = r.recognize_google(audio_data , language=language)
        main_vid = path.split('/')[-2]
        file_trans = f'trans_{main_vid}.txt'
        chunk = path.split('/')[-1].split('.')[0]
        with open(file_trans, 'a') as file_transcript:
            file_transcript.write(f'{chunk} :{text}')
        #print(text)
        gc.collect()
    return text


#print(get_small_audio("D:\Stage de fin d'étude\model-fusion\Landmarks Detector\Audio\Breaking_Bad.wav",language="en-EN"))
