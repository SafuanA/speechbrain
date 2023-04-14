def isLinux():
    if sys.platform == "linux" or sys.platform == "linux2":
        return True
    return False

import os
import numpy as np
import torch
from functools import reduce
import sys
# setting path
#sys.path.append('../speechbrain')
if isLinux():
    sys.path.insert(0,'../../speechbrain')
else:
    sys.path.append('../speechbrain')

from speechbrain.pretrained import SpeakerRecognition

class TransformerSpeakerRecognitionDecoder(SpeakerRecognition):
    # Here, do not hesitate to also add some required modules
    # for further transparency.
    HPARAMS_NEEDED = []#["compute_features", "embedding_model"]
    MODULES_NEEDED = [
         "compute_features",
         "embedding_model",
         "classifier",
         "mean_var_norm",
    ]
    def __init__(self, *args, **kwargs):
         super().__init__(*args, **kwargs)
         # Do whatever is needed here w.r.t your system

    def classify_file(self, path, rel_length = torch.tensor([1.0])):
        out_prob, score, index, text_lab = super().classify_file(path, rel_length)
        #ignore index and score since we use argmax
        index = torch.argmax(out_prob, dim=-1)
        score = out_prob[0,index]
        return out_prob, score, index, text_lab
     

#checkpoint = "CKPT+2023-04-12+03-23-44+00"
src = "C:/Users/saach/source/repos/speechbrain/pre_trained"
#saveDir = "C:/Users/saach/source/repos/speechbrain/pre_trained"

my_model = TransformerSpeakerRecognitionDecoder.from_hparams(
    source=src,
    savedir=src
)

path1 = "C:/Users/saach/source/repos/speechbrain/data/LibriSpeech/train-clean-5/19/198/"
file1 = "19-198-000"

path2 = "C:/Users/saach/source/repos/speechbrain/data/LibriSpeech/train-clean-5/32/21631/"
file2 = "32-21631-000"

suffix = ".flac"

for i in np.arange(10):
        filePath1 = path1 + file1 + str(i) + suffix;
        filePath2 = path2 + file2 + str(i) + suffix;
        _, score, index, label = my_model.classify_file(filePath1, rel_length=None)
        _, score1, index1, label1 = my_model.classify_file(filePath2, rel_length=None)
        print("spk 19:", index, score, label)
        print("spk 32:", index1, score1, label1)

for i in np.arange(10):
    for j in np.arange(10):
        filePath1 = path1 + file1 + str(i) + suffix;
        filePath2 = path2 + file2 + str(j) + suffix;

        filePath3 = path1 + file1 + str(i+1) + suffix;
        res = my_model.verify_files(filePath1, filePath2, normalize=False, threshold=0.9)
        print(res, "not equal")

        res = my_model.verify_files(filePath1, filePath3, normalize=False, threshold=0.9)
        print(res, "equal")



#audio_file = 'your_file.wav'
#encoded = my_model.encode_file(audio_file)
#dec = TransformerSpeakerRecognitionDecoder()
#dec.verify_files()