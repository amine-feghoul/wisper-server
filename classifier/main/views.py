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
from .classifiers import CLASSIFERS



#adding model
uncertanThreshHold = 0.6
loudnessThresHold = 40
new_model = tf.keras.models.load_model('./main/classifier.h5',compile=False)
new_model.compile(loss="categorical_crossentropy",metrics=['accuracy'],optimizer="adam")

#defining the classes 
classes = ['car-horn','dog-bark','human voice']


@api_view(['GET',"POST"])
def predict(request):
    if request.method == "POST":
        try :
            start_time = time.time()
            if len(request.FILES) > 0:
                print(request.FILES)
                up_file = request.FILES['audio']
                if not "audio" in os.listdir("./"):
                    print(os.listdir("./"))
                    destination = os.mkdir('./audio/')
                if 'temp.3gp' in os.listdir("./audio/"):
                    os.remove('./audio/temp.3gp')
                file = open('./audio/temp.3gp',"wb")
                if "temp.wav" in os.listdir("./audio/"):
                    os.remove("./audio/temp.wav")
                for chunk in up_file.chunks():
                    file.write(chunk)
                file.close()
                
                os.system("ffmpeg -i  ./audio/temp.3gp ./audio/temp.wav")        
                
                classifier = request.data.get("classifier")
                
                print("selected classifier is " ,classifier)
                result = CLASSIFERS[classifier].prdict(classes)

                os.remove("./audio/temp.wav")
                
                end_time = time.time()
                print("time elasped is " ,end_time * 1000 - start_time * 1000)
                return JsonResponse({"file":"file found","class":result},status = 200 ,safe=False) 
        except Exception as e:
            print(e)
            return JsonResponse({"err":"error happend",},status = 500 ,safe=False) 
        else:
            return JsonResponse({"file":"file not found"},status = 401 ,safe=False)
    return JsonResponse({"err":"method not recognized"},status = 200 ,safe=False)

@api_view(['GET'])
def getClassifers(request):
    return JsonResponse({"classifers":[ i for i in CLASSIFERS]},status = 200 ,safe=False)