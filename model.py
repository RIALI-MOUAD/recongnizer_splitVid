from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Input, Flatten, Embedding, Convolution1D,Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Input,Dropout,ZeroPadding2D,Conv2D

from tensorflow.keras.layers import Input, Dense, Dropout, Activation, concatenate
import os, re
import numpy as np
import torch
import torchvision
import torch.nn as nn
from transformers import CamembertModel, CamembertConfig, AutoTokenizer
from torch.nn import functional as F
from utils import *
##------------ Audio Modality -----------
class VGG(nn.Module):
    """
    Speech Recognition model with a VGG base
    VGG to latent representation (temporal)
    Aggragation of the temporal representation with mean
    Dense-Dense-SoftMax at the head to classify emotions
    """


    def __init__(self,
                 name='vgg11',
                 bn=True,
                 save_path=f'{os.path.dirname(os.path.abspath(__file__))}/models/vgg_11_final_v1.pth'):
        """
        :param name: name of the vgg base ('vgg11', 'vgg13', 'vgg16' or 'vgg19')
        :param bn: wether or not to use Batch Normalization
        :param save_path: save path to recover weights of the model if given
        """

        dict_vgg = {
            ('vgg11', False): torchvision.models.vgg11,
            ('vgg11', True): torchvision.models.vgg11_bn,
            ('vgg13', False): torchvision.models.vgg13,
            ('vgg13', True): torchvision.models.vgg13_bn,
            ('vgg16', False): torchvision.models.vgg16,
            ('vgg16', True): torchvision.models.vgg16_bn,
            ('vgg19', False): torchvision.models.vgg19,
            ('vgg19', True): torchvision.models.vgg19_bn,
        }

        super(VGG, self).__init__()
        # set the model and recover pretrained weights of the VGG
        # self.vgg = dict_vgg[(name, bn)](pretrained=True, progress=True)
        self.vgg = dict_vgg[(name, bn)](pretrained=False, progress=True)

        # last convolution to get a Bx1xLxT
        # where B the batch size, L the latent size and T the temporal size
        self.conv_mel_to_flat = nn.Conv2d(
            512,
            512,
            kernel_size=(2, 1),
            stride=1,
            dilation=1,
            padding=0
        )


        # Head from latent representation to emotion classification
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 5)
        self.activation = nn.Softmax()

        # set the device by default GPU if exists
        # and recover saved model if given
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if save_path is not None:
            self.load_state_dict(torch.load(save_path, map_location=self.device))

    def predict_sample(self, x):
        """
        Make a prediction for one MEL Spectrogram input

        :param x: MEL spectrogram
        :return: prediction as emotion array
        """
        # create channel axis 1 x Mel x Time
        x = torch.from_numpy(x).unsqueeze(0)

        # assure the input is at least Time > 32 to avoid computing issues
        minimum_t = 32
        if x.size()[-1] < minimum_t:
            c, m, t = x.size()
            _x = torch.zeros((c, m, minimum_t))
            _x[..., :t] = x
            x = _x

        # set the input to the model's device then add the 1 bacth axis
        x = x.to(self.device)
        pred, latent = self.forward(x.unsqueeze(0))

        # get the prediciton and latent to the current cpu device to numpy
        pred = pred[0].detach().cpu().numpy()
        latent = latent[0].detach().cpu().numpy()
        return pred, latent


    def forward(self, x):
        """
        Forward pass of the model

        :param x: MEL spectrogram
        :return: prediction as emotion array
        """
        # convert Batch x 1 x Mel x Time to Batch x 3 x Mel x Time
        # because the model was trained on RGB images
        x = torch.cat([x, x, x], dim=1)

        # get the features Batch x 3 x Latent x Time
        x = self.vgg.features(x)
        # convert Batch x 3 x Latent x Time to Batch x (1) x Latent x Time
        x = self.conv_mel_to_flat(x).squeeze(-2)  # (Batch x Latent x Time)
        # aggregate the latent space with temporal mean
        x = torch.mean(x, dim=2)  # (Batch x Latent)

        latent = x
        # make classifcation from latent space
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.activation(self.fc2(x))  # (Batch x N_class)
        return x, latent
##-----------------Visual Modality -------------------
class Custom_modal:
    def __init__(self,
                 output = 8,
                 path_to_weights=os.path.join(os.path.dirname(__file__),
                 "models/weight_fabo_model_non_verbal9527010920201511.h5")) -> None:
        # create the base pre-trained model
        base_model = ResNet152V2( include_top=False, input_shape=(200, 256, 3))
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(8, activation='softmax')(x)
        self.model = Model(base_model.input, x)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.load_weights(path_to_weights)
        #print(self.model.summary())
    #def pretrained_weights(self,path_to_weights=os.path.join(os.path.dirname(__file__),"Weights/Conv_3_5.h5")): #Weights\Conv_3_5.h5
    def predict_emotions(self, img):
        self.preds = self.model.predict(img)
        self.pourcentage = self.preds/np.sum(self.preds) * 100
        #print('\n',self.preds,'\n')
        #self.pourcentage= np.sort(self.pourcentage)
        return self.pourcentage



# ---------- Text modality ----------
# ---------- Constantes ----------
TOKENIZER = None
CLASSIFIER = None
MAX_LEN = 200
LABELS_TEXT = ['joy', 'neutral', 'sadness', 'anger', 'other']

# Calibration and threshold
SIGMA = 0.6
T = torch.tensor(2.4470)
# ---------- Modèle ----------
class Classifier(nn.Module):
  def __init__(self, init_size):
    super().__init__()
    self.fc1 = nn.Linear(init_size, 100)
    self.fc2 = nn.Linear(100, 5)
    self.T = T

  def forward(self, x):
    out = F.relu(self.fc1(x))
    y_hat = F.softmax(self.fc2(out) / self.T, dim=1)
    return y_hat

class CamembertClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.camembert = CamembertModel(CamembertConfig.from_pretrained('camembert-base'))

    # Classifier
    self.dropout = nn.Dropout(0.5)
    self.classifier = Classifier(self.camembert.config.hidden_size)


  def forward(self, x):
    x_ids, x_am = x['tokens'], x['attention_masks']

    out = self.camembert(
        input_ids=x_ids,
        attention_mask=x_am
    )
    out = self.dropout(out[1])
    y_hat = self.classifier(out)
    return y_hat


# ---------- Preprocessing ----------
# Text normalisation
punctuation_translations = [
    (r'\.\.', '...'),
    (r'\.\.\.[\.]*', '...'),
    (r'[ ]+([\.,\']+)', r'\1'),
    (r'[ ]([\?!;:]+)', r'\1'),
    (r'([\']+)[ ]+', r'\1'),
    (r'([\?\.!,;:]+)([^ ])', r'\1 \2'),
    (r'([\?\.!,;:]+)[ ]+([\?\.!,;:]+)', r'\1\2'),
    (r'![!]+', '!!'),
    (r'\?[\?]+', '??'),
]
def norm_punctuation(text):
  for k,v in punctuation_translations:
    text = re.sub(k, v, text)
  return text

def norm_text(text):
  return text.lower()

# Tokenisation
PAD_TOKEN = None
CLS_TOKEN = None
SEP_TOKEN = None
def tokenize(utterances):
  encodings = [TOKENIZER.encode_plus(u,
                                    add_special_tokens=False,
                                    max_length=MAX_LEN,
                                    return_attention_mask=False,
                                    truncation=True)['input_ids']
                                    for u in utterances if u is not None and u != '']
  if len(encodings) > 0:
    encodings[0].insert(0, CLS_TOKEN)
  for e in encodings:
    e.append(SEP_TOKEN)
  tokens = torch.from_numpy(np.concatenate(encodings))

  if len(tokens) >= MAX_LEN:
    tokens = tokens[0:MAX_LEN]
  else:
    tokens = F.pad(tokens, pad=(0, MAX_LEN - len(tokens)), value=PAD_TOKEN)
  attention_mask = (tokens != PAD_TOKEN).int()

  return {
      'tokens': tokens,
      'attention_mask': attention_mask,
  }

def preprocess(utterances):
  utterances = [norm_punctuation(norm_text(utterances[0]))]

  tokenization = tokenize(utterances)

  return {
    'tokens': torch.unsqueeze(tokenization['tokens'], 0),
    'attention_masks': torch.unsqueeze(tokenization['attention_mask'], 0)
  }

def round(f):
  return f'{f:0.2f}'

# ---------- Utilitaires ----------
'''
Utilisation du GPU si disponible
'''
gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
def to_device(obj, dev):
  if not isinstance(obj, dict):
    return obj.to(dev)
  else:
    for k in obj:
      obj[k] = to_device(obj[k], dev)
    return obj

def to_gpu(obj):
  return to_device(obj, gpu)
def to_cpu(obj):
  return to_device(obj, cpu)

# ---------- Traitements complets ----------
def initialize():
  global TOKENIZER
  global CLASSIFIER
  global PAD_TOKEN
  global CLS_TOKEN
  global SEP_TOKEN

  #print('### Emotion Initialization ###')
  #print('Tokenizer')
  TOKENIZER = AutoTokenizer.from_pretrained('camembert-base')
  PAD_TOKEN = TOKENIZER.pad_token_id
  CLS_TOKEN = TOKENIZER.cls_token_id
  SEP_TOKEN = TOKENIZER.sep_token_id

  #print('Classifier')
  CLASSIFIER = CamembertClassifier()
  CLASSIFIER.load_state_dict(torch.load(f'{os.path.dirname(os.path.abspath(__file__))}/models/camembert-classifier.pt', map_location='cpu'))
  CLASSIFIER = to_gpu(CLASSIFIER)
  CLASSIFIER.eval()

def predict(utterances):
  _ = initialize()
  x = preprocess(utterances)
  x = to_gpu(x)

  y_hat = CLASSIFIER(x)
  y_hat = to_cpu(y_hat.squeeze())

  # Threshold sigma=0.6
  emotion_label = LABELS_TEXT[1]
  if y_hat.max() >= SIGMA:
    emotion_label = LABELS_TEXT[torch.argmax(y_hat)]

  return {LABELS_TEXT[i]:float("{:.2f}".format(y_hat[i].item()*100)) for i in range(len(LABELS_TEXT))}
    #'emotion': emotion_label,
