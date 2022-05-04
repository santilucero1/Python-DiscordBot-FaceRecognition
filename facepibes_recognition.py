from cProfile import label
from importlib.resources import path
from cv2 import imread, imshow
from matplotlib import image
import numpy as np
import cv2 as cv 
import os
import urllib.request






p = []

for i in os.listdir(r'path to the train folder'):#Example C:\Opencv Test\trainpibes
    p.append(i)

#instance haarcascade
haar_cascade = cv.CascadeClassifier("haar_face.xml")
#instance face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("trainedfile.yml")#Example face_pibes_trained.yml

def url_to_image(url):
    
    #The mod_security is usually configured in such a way that if any requests happen without a valid user-agent header(browser user-agent),
    #the mod_security will block the request and return the urllib.error.httperror: http error 403: forbidden
    #so you can use an opener like this
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36')
    filename, headers = opener.retrieve(url, "save_the_image_in_this_path") #Example ./garbagefolder/img.jpg
    
    
    img = cv.imread(r'path to the image') #Example C:\Opencv Test\garbagefolder\img.jpg
    
    #Convert to gray
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #Detect face of this image with haar_cascade
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 7)
    
    
    for (x,y,w,h) in faces_rect:
        #Get region of interest
        faces_roi = gray[y:y+h,x:x+w]
        #Predict
        label, confidence = face_recognizer.predict(faces_roi)
        #Draw a rectangle and the label on the img
        cv.putText(img, str(p[label]), (x,y-20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        #Save this image
        cv.imwrite('path to save', img) #Example C:\Opencv Test\garbagefolder\imgrecogniced.jpg
        #print label and confidence (its not necesary)
        print(f'Label = {p[label]} with a confidence of {confidence}')
        #Return label (name) and confidence value
        return p[label], confidence

    

    cv.waitKey(0)



