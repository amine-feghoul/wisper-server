import os
from django.shortcuts import render
from rest_framework.decorators import api_view
from django.http import request, HttpResponse, JsonResponse
import librosa
import numpy as np
import tensorflow as tf
import time
import scipy as sc
import matplotlib.pyplot as plt
import subprocess
from scipy.signal import butter, filtfilt


# Define filter parameters
cutoff_freq = 1000 # Hz
sampling_rate = 22050 # Hz
order = 10 # Order of the filter

#adding model
uncertanThreshHold = 0.6
loudnessThresHold = 70
new_model = tf.keras.models.load_model('./main/classifier.h5',compile=False)
new_model.compile(loss="categorical_crossentropy",metrics=['accuracy'],optimizer="adam")

#defining the classes 
classes = ['car-horn','dog-bark','human voice']


# declaring butterworth filter 

# Normalize cutoff frequency
normalized_cutoff_freq = cutoff_freq / (sampling_rate / 2)

# Create a butterworth filter
b, a = butter(order, normalized_cutoff_freq, btype='low', analog=False)



def feature_extractor():
    file,sr =librosa.load("./audio/temp.wav")
    filtered_audio = filtfilt(b, a, file)
    fft =  sc.fft.fft(filtered_audio)
    magnitude = np.absolute(fft)
    
    if(np.max(magnitude) < loudnessThresHold):
        return None
    mfccs = librosa.feature.mfcc(y=filtered_audio, sr=sr, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T,axis=0)
    return mfccsscaled

def get_class(model):
    mfcc = feature_extractor()  
    if mfcc is not None:  
        mfcc = mfcc.reshape(1,-1)
        pred = model.predict(mfcc)
        if(np.max(pred)<uncertanThreshHold):
            return "not sure"
        result = classes[np.argmax(pred)]
        print(result)
        print(pred)
        return result
    else :
        return "silence"
@api_view(['GET',"POST"])
def predict(request):
    if request.method == "POST":
        start_time = time.time()
        if len(request.FILES) > 0:
            print(request.FILES)
            
            up_file = request.FILES['audio']
            if not "audio" in os.listdir("./"):
                print(os.listdir("./"))
                print('entered')
                destination = os.mkdir('./audio/')
            file = open('./audio/temp.3gp',"wb")

            for chunk in up_file.chunks():
                file.write(chunk)
            file.close()
            os.system("y |ffmpeg -i  ./audio/temp.3gp ./audio/temp.wav")        
           
            result = get_class(new_model)
            os.remove("./audio/temp.wav")
            end_time = time.time()
            print("time elasped is " ,end_time * 1000 - start_time * 1000)
            return JsonResponse({"file":"file found","class":result},status = 200 ,safe=False) 
        else:
            return JsonResponse({"file":"file not found"},status = 401 ,safe=False)
    return JsonResponse({"class":"dog bark"},status = 200 ,safe=False)