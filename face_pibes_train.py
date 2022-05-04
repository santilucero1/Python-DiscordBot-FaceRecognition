from importlib.resources import path
import os
from xml.sax.handler import feature_external_ges
from xml.sax.saxutils import prepare_input_source 
import cv2 as cv
from matplotlib.pyplot import gray
import numpy as np

p = []

for i in os.listdir(r'C:\Opencv Test\trainpibes'):
    p.append(i)

print(p)

DIR = r"C:\Opencv Test\trainpibes"

haar_cascade = cv.CascadeClassifier("haar_face.xml")

feautures = []
labels = []

def create_train():
    #get the path of folders
    for person in p:
        path = os.path.join(DIR, person)
        label = p.index(person)
        
        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            

            # #get position of the faces
            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
            
            for (x,y,w,h) in faces_rect:
                #faces regions of interest
                faces_roi = gray[y:y+h, x:x+w]
                #add to features
                feautures.append(faces_roi)
                labels.append(label)

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
face_recognizer.save("face_pibes_trained.yml")

cv.waitKey(0)