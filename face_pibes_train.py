from importlib.resources import path
import os
from xml.sax.handler import feature_external_ges
from xml.sax.saxutils import prepare_input_source 
import cv2 as cv
from matplotlib.pyplot import gray
import numpy as np

#!!!
#Firs at all you have to download the haarcascade file here: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml


#Get the names of the "peoples folders" with the pics in the "Train folder"
p = []
for i in os.listdir(r'path'):#Example C:\Opencv Test\trainpibes
    p.append(i)
print(p)

#Dir to Train Folder
DIR = r'path' #Example "C:\Opencv Test\trainpibes"

#Instance the haarcascade
haar_cascade = cv.CascadeClassifier("haar_face.xml")

#Empty list of feautures and labels
feautures = []
labels = []

def create_train():
    #get the path of folders
    for person in p:
        path = os.path.join(DIR, person)
        label = p.index(person)
        
        #get the path of the imgs
        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
            
            #Trasnform to gray
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            

            #Get position of the faces
            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
            
            for (x,y,w,h) in faces_rect:
                #faces regions of interest
                faces_roi = gray[y:y+h, x:x+w]
                #add to features
                feautures.append(faces_roi)
                labels.append(label)

#Execute the train
create_train()
print("Training done-----------")

print(f"length of the features = {len(feautures)}")
print(f"length of the labels = {len(labels)}")

#need to convert the features and labels to np arrays
feautures=np.array(feautures, dtype="object")
labels=np.array(labels)


#Instanciate of recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
#We can use features and labels to train our recognizer
face_recognizer.train(feautures,labels)

#Save features and labels
np.save("featurespibes.npy", feautures)
np.save("Labelspibes.npy",labels)
#Can save this training mode in anothe file using yml
face_recognizer.save("face_pibes_trained.yml")#save the result of the train to the  "something.yml" file

cv.waitKey(0)
