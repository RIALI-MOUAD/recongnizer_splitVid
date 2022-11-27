from read_frames import *
from split_vid import *
from utils import *
from extract_data import *
from predict import *



if __name__== '__main__':
    main_vid = 'We_all_feel'
    language = 'en-EN'
    main_split(main_vid, 0.15)
    #fps = main_predict(main_vid = main_vid, language=language)
    chunks_vid = glob.glob(f'frames/{main_vid}/chunk*')
    chunks_vid.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    all_frames(chunks=chunks,main_vid = main_vid)
    frames_glob = get_all_frames(main_vid = main_vid, chunks=chunks_vid)
    print(len(frames_glob))
    file_vid = f'Videos_out/video_test_{main_vid}.mp4'
    write_video(file_path=file_vid, frames=frames_glob, fps=25)
