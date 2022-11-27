from pydub import AudioSegment
from pydub.silence import split_on_silence
import moviepy.editor as mvp
from moviepy.editor import VideoFileClip
import os
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import math
import sys
import subprocess
import shutil
from moviepy.editor import AudioClip, VideoFileClip, concatenate_videoclips
from utils import find_speaking
import glob

def split_audio(video='comment_5', min_silence_len=500, silence_thresh=-40):
    audio_clip = mvp.AudioFileClip(f'Videos/{video}.mp4')
    audio_clip.write_audiofile(f'Audios/{video}.wav')
    sound_file = AudioSegment.from_wav(f'Audios/{video}.wav')
    audio_chunks = split_on_silence(sound_file, min_silence_len, silence_thresh)
    directory = f'Audio_clips/{video}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    L = [0]
    for i, chunk in enumerate(audio_chunks):
        out_file = directory+"/chunk{0}.wav".format(i)
        print("exporting", out_file)
        chunk.export(out_file, format="wav")
        L.append(chunk.duration_seconds)
    return np.cumsum(L)

def convert_audio(path_1,path_2,ext1, ext2):
    track = AudioSegment.from_file(path_1,  format= ext1 )
    file_handle = track.export(path_2, format=ext2)


def trim_vid(video='comment_5', start_time=0, end_time=1.86698413,indx =0):
    directory = f'Video_clips/{video}'
    directoryAudio = f'Audio_clips/{video}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directoryAudio):
        os.makedirs(directoryAudio)
    try:
        clip = VideoFileClip(f'Videos/{video}.mp4').subclip(abs(start_time), end_time)
        clip.to_videofile(f'{directory}/chunk{indx}.mp4', codec="libx264", temp_audiofile=f'{directoryAudio}/chunk{indx}.m4a',
         remove_temp=False, audio_codec='aac')
    except:
        print(start_time, end_time)



def split_video(video = 'zAjJYkUnTEs',volume_threshold=0.15):
    clip = VideoFileClip(f'Videos/{video}.mp4')
    intervals_to_keep = find_speaking(clip.audio,volume_threshold=volume_threshold)
    print(len(intervals_to_keep))
    i = 0
    for periode in intervals_to_keep:
        trim_vid(video, periode[0], periode[1],i )
        i+=1



def main_split(video = 'zAjJYkUnTEs',volume_threshold=0.15):
    split_video(video = video,volume_threshold= volume_threshold)
    files = glob.glob(f'Audio_clips/{video}/chunk*.m4a')
    for file in files:
        path_2 = file.split('.')[0]+'.wav'
        ext1, ext2 = 'm4a', 'wav'
        convert_audio(file,path_2,ext1, ext2)
