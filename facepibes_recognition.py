from cProfile import label
from importlib.resources import path
from cv2 import imread, imshow
from matplotlib import image
import numpy as np
import cv2 as cv 
import os
import urllib.request






p = []

for i in os.listdir(r'C:\Opencv Test\trainpibes'):
    p.append(i)

haar_cascade = cv.CascadeClassifier("haar_face.xml")

# features = np.load("features.npy")
# labels = np.load("labels.npy")
# i can use these data say ,allow_pickle=True

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_pibes_trained.yml")

def url_to_image(url):
    
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36')
    filename, headers = opener.retrieve(url, "./garbagefolder/img.jpg")
    img = cv.imread(r'C:\Opencv Test\garbagefolder\img.jpg')
    cv.imshow("image",img)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imshow("Person",gray)
    
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 7)
    
    
    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        cv.putText(img, str(p[label]), (x,y-20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        cv.imwrite('C:\Opencv Test\garbagefolder\imgrecogniced.jpg', img) 
        #print label and confidence
        print(f'Label = {p[label]} with a confidence of {confidence}')
        return p[label], confidence

    

    cv.waitKey(0)



