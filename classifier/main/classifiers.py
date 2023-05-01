import numpy as np
import librosa
import librosa
import numpy as np
import tensorflow as tf
import time
import scipy as sc
import matplotlib.pyplot as plt
import subprocess
from scipy.signal import butter, filtfilt

loudnessThresHold = 40

# Define filter parameters
cutoff_freq = 5000 # Hz
sampling_rate = 44100 # Hz
order = 2 # Order of the filter


# declaring butterworth filter 

# Normalize cutoff frequency
normalized_cutoff_freq = cutoff_freq / (sampling_rate / 2)

# Create a butterworth filter
b, a = butter(order, normalized_cutoff_freq, btype='low', analog=False)





def ann_feature_extractor():
    file,sr =librosa.load("./audio/temp.wav")
    filtered_audio = filtfilt(b, a, file)
    fft =  sc.fft.fft(filtered_audio)
    magnitude = np.absolute(fft)
    
    if(np.max(magnitude) < loudnessThresHold):
        return None
    mfccs = librosa.feature.mfcc(y=filtered_audio, sr=sr, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T,axis=0)
    return mfccsscaled

def cnn_feature_extractor(file):
    max_pad_len = 174
    audio, sample_rate = librosa.load("./audio/temp.wav")                  

    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
     
    pad_width = max_pad_len - mfccs_features.shape[1]
    if pad_width < 0 :
        mfccs_aug = mfccs_aug[:,:174]
        mfccs_features = mfccs_features[:,:174]
    else:
        mfccs_features = np.pad(mfccs_features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        mfccs_aug= np.pad(mfccs_aug, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs_features

class Classifer:
    def __init__(self,model,feature_extract,uncertanThreshHold = 0):
        self.model = model
        self.feature_extractor = feature_extract
        self.uncertanThreshHold = uncertanThreshHold
    def prdict(self,classes):
        mfcc = self.feature_extractor() 
        if mfcc is not None:  
            mfcc = mfcc.reshape(1,-1)
            pred = self.model.predict(mfcc)
            if(np.max(pred)<self.uncertanThreshHold):
                return "not sure"
            result = classes[np.argmax(pred)]
            print(result)
            print(pred)
            return result 
        return "silence" 
annModel = tf.keras.models.load_model('./main/classifier.h5',compile=False)
annModel.compile(loss="categorical_crossentropy",metrics=['accuracy'],optimizer="adam")
ANN = Classifer(model=annModel,feature_extract=ann_feature_extractor,uncertanThreshHold=0.6)

## CNN model needs to be imported and the feature extractor for cnn also 
cnnModel = tf.keras.models.load_model('./main/cnn.h5',compile=False)
cnnModel.compile(loss="categorical_crossentropy",metrics=['accuracy'],optimizer="adam")
CNN = Classifer(model=cnnModel,feature_extract=cnn_feature_extractor,uncertanThreshHold=0.6)

CLASSIFERS = {"ANN":ANN,"CNN":CNN}