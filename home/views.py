# from multiprocessing import context
# from django.http import HttpResponse
from django.shortcuts import render, redirect

from .models import *
from django.core.files.storage import FileSystemStorage

import tensorflow as tf
from tensorflow import keras
from scipy.spatial import distance

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import io
import urllib, base64

import cv2 
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import os
import json

img_height, img_width=128, 128

model = tf.keras.models.load_model('C:/Users/LENOVO/Documents/skripsi/program/web2/model/VGG19-Face Mask Detection.h5')

face_model = cv2.CascadeClassifier('C:/Users/LENOVO/Documents/skripsi/program/web2/haarcascades/haarcascade_frontalface_default.xml')

mask_label = {0:'MASK',1:'NO MASK'}
dist_label = {0:(0,255,0),1:(255,0,0)} # rectangle color


def index(request):
    context={'a':1}
    return render(request, 'index.html', context)

def about(request):
    context={'a':1}
    return render(request, 'about.html', context)

def plot_image(image,subplot):
    plt.subplot(*subplot)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show

def predictImage(request):

    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    image_dir = '.'+filePathName

    img = cv2.imread(image_dir)
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    
    #returns a list of (x,y,w,h) tuples
    faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    
    detection = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # plt.figure(figsize=(10,10))
    # plot_image(detection,(1,2,1))
    
    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        crop = detection[y:y+h,x:x+w]
        crop = cv2.resize(crop,(128,128))
        crop = np.reshape(crop,[1,128,128,3])/255.0
        mask_result = model.predict(crop).argmax()
        cv2.rectangle(detection,(x,y),(x+w,y+h),dist_label[mask_result],1)
    
    # plot_image(detection,(1,2,2))
    plt.imsave('C:/Users/LENOVO/Documents/skripsi/program/web2/facemask/media/detection.png', detection)

    context={'detection':detection, 'filePathName':filePathName}
    return render(request, 'face_mask_detection.html', context)

def detection_face(request):

    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    image_dir = '.'+filePathName

        # new_img = np.array(image.convert('RGB'))
    new_img =cv2.imread(image_dir)

    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Detect Faces
    faces = face_model.detectMultiScale(gray,1.3,4)
        #draw Rectangle 
    for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)

    cv2.imwrite('C:/Users/LENOVO/Documents/skripsi/program/web2/facemask/media/img.png', img)

    context={'img':img, 'faces':faces, 'filePathName':filePathName}
    return render(request, 'face_detection.html', context)

def face_detection(request):
    context={'a':1}
    return render(request, 'face_detection.html', context)

def face_mask_detection(request):
    context={'a':1}
    return render(request, 'face_mask_detection.html', context)

def dataset(request):

    image_size = (128,128)

    base_dir = 'C:/Users/LENOVO/Documents/skripsi/program/input/face-mask-12k-images-dataset/Face Mask Dataset/'


    train_gen = keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True,
            rescale=1./255
        )

    val_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_ds = train_gen.flow_from_directory(base_dir+'Train',target_size=image_size,seed=42)
    val_ds = val_gen.flow_from_directory(base_dir+'Validation',target_size=image_size,seed=42)
    test_ds = test_gen.flow_from_directory(base_dir+'Test',target_size=image_size,seed=42)

    class_names = {v:k for k,v in train_ds.class_indices.items()}
    images,labels = next(iter(train_ds))

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
            # ax.hist(arr, bins=20)
        plt.imshow(images[i])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(class_names[labels[i][1]])

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())

    uri = 'data:image/png;base64,' + urllib.parse.quote(string)

    context={'train_ds':train_ds, 'val_ds':val_ds, 'test_ds':test_ds, 'image':uri}
    return render(request, 'dataset.html', context)

def accouracy(request):

    # image = Image.open('C:/Users/LENOVO/Documents/skripsi/hasil dan pembahasan/download.png')

    context={'a':1}
    return render(request, 'accouracy.html', context)



# def index(request):
#     if request.method == 'POST' and request.FILES['upload']:
#         upload = request.FILES['upload']
#         fss = FileSystemStorage()
#         file = fss.save(upload.name, upload)
#         file_url = fss.url(file)
#         return render(request, 'index.html', {'file_url': file_url})

#     return render(request, 'index.html')

# import dlib

# detector = cv2.CascadeClassifier('C:/Users/LENOVO/Documents/skripsi/program/web2/haarcascades/haarcascade_frontalface_default.xml')
# new_path ='C:/Users/LENOVO/Documents/skripsi/program/input/face-mask-12k-images-dataset/Face Mask Dataset/Test/WithoutMask'

# def MyRec(rgb,x,y,w,h,v=20,color=(200,0,0),thikness =2):
#     """To draw stylish rectangle around the objects"""
#     cv2.line(rgb, (x,y),(x+v,y), color, thikness)
#     cv2.line(rgb, (x,y),(x,y+v), color, thikness)

#     cv2.line(rgb, (x+w,y),(x+w-v,y), color, thikness)
#     cv2.line(rgb, (x+w,y),(x+w,y+v), color, thikness)

#     cv2.line(rgb, (x,y+h),(x,y+h-v), color, thikness)
#     cv2.line(rgb, (x,y+h),(x+v,y+h), color, thikness)

#     cv2.line(rgb, (x+w,y+h),(x+w,y+h-v), color, thikness)
#     cv2.line(rgb, (x+w,y+h),(x+w-v,y+h), color, thikness)

# def save(img,name, bbox, width=180,height=227):
#     x, y, w, h = bbox
#     imgCrop = img[y:h, x: w]
#     imgCrop = cv2.resize(imgCrop, (width, height))#we need this line to reshape the images
#     cv2.imwrite(name+".png", imgCrop)

# def faces():
#     new_path ='C:/Users/LENOVO/Documents/skripsi/program/input/face-mask-12k-images-dataset/Face Mask Dataset/Test/WithoutMask'

#     frame =cv2.imread(new_path)
#     gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     fit =20
#     # detect the face
#     for counter,face in enumerate(faces):
#         print(counter)
#         x1, y1 = face.left(), face.top()
#         x2, y2 = face.right(), face.bottom()
#         cv2.rectangle(frame,(x1,y1),(x2,y2),(220,255,220),1)
#         MyRec(frame, x1, y1, x2 - x1, y2 - y1, 10, (0,250,0), 3)
#         # save(gray,new_path+str(counter),(x1-fit,y1-fit,x2+fit,y2+fit))
#         save(gray,new_path+str(counter),(x1,y1,x2,y2))
#     frame = cv2.resize(frame,(800,800))
#     cv2.imshow('img',frame)
#     cv2.waitKey(0)
#     print("done saving")

# faces()