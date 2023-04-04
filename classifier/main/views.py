import os
from django.shortcuts import render
from rest_framework.decorators import api_view
from django.http import request, HttpResponse, JsonResponse
import librosa
import numpy as np
import tensorflow as tf
# Create your views here.

new_model = tf.keras.models.load_model('./main/classifier.h5')

classes = ['car-horn','dog-bark']
def feature_extractor():
    file,sr =librosa.load("./audio/temp.wav")
    mfccs = librosa.feature.mfcc(y=file, sr=sr, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T,axis=0)
    return mfccsscaled


def get_class(model):
    mfcc = feature_extractor()
    mfcc = mfcc.reshape(1,-1)
    pred = model.predict(mfcc)
    result = classes[np.argmax(pred)]
    print(result)
    return result

@api_view(['GET',"POST"])
def predict(request):
    if request.method == "POST":
        if len(request.FILES) > 0:
            print(request.FILES)
            up_file = request.FILES['audio']
            if not "audio" in os.listdir("./"):
                print(os.listdir("./"))
                print('entered')
                destination = os.mkdir('./audio/')
            file = open('./audio/temp.wav',"wb")
            for chunk in up_file.chunks():
                file.write(chunk)
            file.close()
            result = get_class(new_model)
            return JsonResponse({"file":"file found","class":result},status = 200 ,safe=False) 
        else:
            return JsonResponse({"file":"file not found"},status = 401 ,safe=False)
    return JsonResponse({"class":"dog bark"},status = 200 ,safe=False)