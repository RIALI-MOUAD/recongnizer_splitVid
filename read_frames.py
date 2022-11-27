#from extract_data import Extract_Data
from werkzeug.utils import secure_filename
import gc
import os, json
import numpy as np
from datetime import timedelta
#from utils import *
import cv2
import matplotlib.pyplot as plt
import time
import glob
from tqdm import tqdm


#--------------
chunks = glob.glob('frames/zAjJYkUnTEs/chunk*')
chunks.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

def get_all_frames(main_vid, chunks=chunks) -> None:
    frames_glob = []
    for i in range(len(chunks)):
        frames_loc = write_on_frames(main_vid = main_vid,chunk = i)
        frames_glob+=frames_loc
    return frames_glob
#-------------

def frames_chunk(path_vid='Video_clips/zAjJYkUnTEs/chunk7.mp4'):
    main_vid = path_vid.split('/')[-2]
    sub_vid = path_vid.split('/')[-1].split('.')[0]
    if not os.path.exists(f'frames/{main_vid}'):
        os.makedirs(f'frames/{main_vid}')
        os.makedirs(f'frames/{main_vid}/{sub_vid}')
    elif not os.path.exists(f'frames/{main_vid}/{sub_vid}'):
        os.makedirs(f'frames/{main_vid}/{sub_vid}')
    cap = cv2.VideoCapture(path_vid)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)

    # Start time
    start = time.time()

    for i in range(length):
        success,image = cap.read()
        if success:
            imageR = image
            path2fr = f'frames/{main_vid}/{sub_vid}/frame{i}.jpg'
            cv2.imwrite(path2fr, imageR)
        else:
            cv2.imwrite(f'frames/{main_vid}/{sub_vid}/frame{i}_fin.jpg', imageR)
            #continue
    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    print ("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    fps  = length / seconds
    print("Estimated frames per second : {0}".format(fps))
    cap.release()

#Extract all frames
def all_frames(main_vid, chunks = chunks) -> None:
    chunks = glob.glob(f'Video_clips/{main_vid}/chunk*.mp4')
    for chunk in chunks:
        frames_chunk(chunk)

#Get emotions
def get_stri(i,main_vid,json_file='sample.json'):
    with open(json_file,'r') as json_file:
        data = json.load(json_file)
    chunks = glob.glob(f'frames/{main_vid}/chunk*')
    indx = chunks[i].split('chunk')[-1]
    data[indx]
    elmnt = data[indx]['N.B']
    try:
        audio, text, video, All = elmnt['Audio'], elmnt['Text'], elmnt['Video'], \
                                  elmnt['Audio']
        return f'Audio :{audio} | Text :{text} | Video :{video}| Most powerful emotion: {All}'
    except:
        return 'Not Detected :)'


#Write on frames
def write_on_frame(path,
                   out,
                   text,
                   font = cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale = 0.5,
                   fontColor = (245, 0, 0),
                   thickness = 2,
                   lineType = 2):
    img = cv2.imread(path)
    shape = img.shape
    # Write some Text
    texts = text.split('|')
    bottomLeftCornerOfText = (shape[1]//40,shape[0]//10)
    dy = 0
    for i in range(len(texts)):
        cv2.putText(img,texts[i],
                (shape[1]//40,shape[0]//10 + dy),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
        dy += shape[0]//30
    #Save image
    cv2.imwrite(out, img)
    cv2.waitKey(0)

#------------------------

def write_on_frames(main_vid,chunk,portion = 0.5, tresh = 45):
    frames = glob.glob(f'frames/{main_vid}/chunk{chunk}/frame*')
    frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    n = min(len(frames),tresh)
    file_sample = f'rapports/sample_{main_vid}.json'
    text = get_stri(main_vid = main_vid,i = chunk,json_file=file_sample)
    for frame in frames[len(frames)-n:]:
        write_on_frame(frame,frame,text)
    return frames


#-----------------
def write_video(file_path, frames, fps=24):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """
    print(frames[0])
    img0 = cv2.imread(frames[0])
    w, h = img0.shape[1],img0.shape[0]
    print(w,h)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame,i in zip(frames,tqdm(range(len(frames)))):
        frame = cv2.imread(frame)
        writer.write(frame)
    writer.release()
