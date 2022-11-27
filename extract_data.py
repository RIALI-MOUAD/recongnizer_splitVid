#from turtle import color
from cProfile import label
from genericpath import exists
import cv2
#from preprocess import preprocess_Audio
from utils import *
import numpy as np
from progressbar import progressbar
import moviepy.editor as mvp
from deep_translator import GoogleTranslator
from model import *
from Audios.audio_convertor import get_small_audio
from sklearn import preprocessing
from model import Custom_modal, VGG


class Extract_Data:
    def __init__(self,path,main_vid) -> None:
        self.video = path.split('/')[-1].split('.')[0]
        self.path_video = path
        self.path_audio = os.path.join(os.path.dirname(__file__)+"/Audio_clips/"+main_vid,self.video+".wav")
        self.main_vid = main_vid
#    def get_Audio(self):
#        audioclip = mvp.AudioFileClip(self.path_video)
#        audioclip.write_audiofile(self.path_audio)
#        print('Done')
#        return process_audio(self.path_audio)
    def get_trans(self, source='en',dest = 'fr',text=""):
        """extracting, translating and processing transcriptions

        Args:
            source (str, optional): _description_. Defaults to 'fr'.
            dest (str, optional): _description_. Defaults to 'en'.
            text (str, optional): _description_. Defaults to "".

        Returns:
            _type_: _description_
        """
        #f = open(self.path_text,'r')
        #text = f.read()
        if source == "en":
            text_fr = GoogleTranslator(source=source, target=dest).translate(text)
            return text_fr
        return text
    def predict_emos_Alteca(self,
                            emotions = ['Anger','Boredom','Disgust','Fear','Happiness','Sadness','Surprise','UNCERTAINTY'],
                            language="en-EN"):
        ## Visual modality
        obj_v, obj_a = Custom_modal(),VGG()
        if not os.path.exists(f'{os.path.dirname(__file__)}/records/{self.main_vid}'):
            os.makedirs(f'{os.path.dirname(__file__)}/records/{self.main_vid}')
        file_record = f'{os.path.dirname(__file__)}/records/{self.main_vid}/{self.video}.npy'
        if exists(file_record):
            S = np.load(file_record,allow_pickle=True)
        else :
            S ,fps= choose_vid(self.path_video,obj=obj_v)
            with open(file_record, 'wb') as f:
                np.save(f, S)
        #print(S)
        glbl = np.mean(S[1:], axis=0)
        dictio_v = {}
        for i in range(len(emotions)):
            #print(emotions[i])
            dictio_v[emotions[i]]=float("{:.2f}".format(glbl[i]))
        ## Speech modality
        #spec = self.get_Audio()
        spec = process_audio(self.path_audio)
        pred_a,_= obj_a.predict_sample(spec)
        dictio_a = {}
        labels_Audio = ['angry','happy', 'neutral','sad', 'other']
        for i in range(len(labels_Audio)):
          dictio_a[labels_Audio[i]]=float("{:.2f}".format(pred_a[i]*100))
        ## Text modality
        transcript = get_small_audio(self.path_audio,language=language)
        text = self.get_trans(source=language[:2],text=transcript)
        #print(text)
        dictio_t = predict([text])
        Shared_Emotions = ['happy','joy','Happiness','sad','sadness','angry','anger','neutral','other']
        others = [dictio_a[key] for key in dictio_a.keys() if key.lower() not in Shared_Emotions]+[dictio_t[key] for key in dictio_t.keys() if key.lower() not in Shared_Emotions]+[dictio_v[key] for key in dictio_t.keys() if key.lower() not in Shared_Emotions]
        #print(others)
        dictio_all = {
            'happiness': float("{:.2f}".format(np.mean(np.array([dictio_a['happy'],dictio_t['joy'],dictio_v['Happiness']])))),
            'sadness': float("{:.2f}".format(np.mean(np.array([dictio_a['sad'],dictio_t['sadness'],dictio_v['Sadness']])))),
            'anger': float("{:.2f}".format(np.mean(np.array([dictio_a['angry'],dictio_t['anger'],dictio_v['Anger']])))),
            'neutral': float("{:.2f}".format(np.mean(np.array([dictio_a['neutral'],dictio_t['neutral'],dictio_t['neutral']]))))
            }
        dictio_all['others'] = 100-dictio_all['happiness']-dictio_all['sadness']-dictio_all['anger']-dictio_all['neutral']
        dictio_max = {
            "Audio": list(dictio_a.keys())[list(dictio_a.values()).index(max(dictio_a.values()))],
            "Text": list(dictio_t.keys())[list(dictio_t.values()).index(max(dictio_t.values()))],
            "Video": list(dictio_v.keys())[list(dictio_v.values()).index(max(dictio_v.values()))],
            "All": list(dictio_all.keys())[list(dictio_all.values()).index(max(dictio_all.values()))]
            }

        dictio = {}
        dictio['Audio'], dictio['Video'],dictio['Text'],dictio['All']=dictio_a, dictio_v, dictio_t,dictio_all
        dictio["N.B"] = dictio_max
        #print(dictio)
        return S,glbl,dictio,fps



#obj = Extract_Data('0h-zjBukYpk_Stripped')
#obj.get_Audio()
#obj.get_Mocap()
#obj.pred_emos()
