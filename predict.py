from extract_data import Extract_Data
from werkzeug.utils import secure_filename
import gc
import os, json
import numpy as np
from datetime import timedelta
from utils import *
import glob
from tqdm import tqdm

def predict(path,main_vid,language="en-EN"): #self,path,main_vid
    ''''
    try :
        #output = int(request.args.get('output'))
    except:
        output = 5
    '''
    obj = Extract_Data(path,main_vid)
    _,_,dictio,fps = obj.predict_emos_Alteca(language=language)
    #print(dictio)
    gc.collect()
    #return jsonify(dictio)
    return dictio,fps#,json.dumps(dictio)


def main_predict(main_vid = 'zAjJYkUnTEs', language="en-EN") -> None:
    #predict('Video_clips/zAjJYkUnTEs/chunk13.mp4','zAjJYkUnTEs')
    files = glob.glob(f'Video_clips/{main_vid}/*.mp4')
    #print(files)
    dict_All = {}
    for file,i in tqdm(zip(files,range(len(files))), total=len(files)):
        dictio_local = {}
        try:
            NB,fps = predict(file, main_vid,language=language)
            NB = NB['N.B']
            #print(NB)
        except:
            NB = 'Not Detected'
        #print(NB)
        i = int(file.split('chunk')[-1].split('.')[0])
        dictio_local['N.B']=NB
        dict_All[i]=dictio_local
    jsonFile = f'rapports/sample_{main_vid}.json'
    with open(jsonFile, "w") as outfile:
        json.dump(dict_All, outfile)
    return fps

"""
"""
