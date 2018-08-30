# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:25:22 2018

@author: prant
"""

from keras.models import load_model
import cv2
import numpy as np
import dlib  
import imutils
import time
import datetime

model = load_model('lefteyemodel.h5')
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
cascPath = 'haarcascade_frontalface_default.xml'  
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

LEFT_EYEBROW_POINTS = list(range(22, 27)) 
LEFT_EYE_POINTS = list(range(42, 48))

faceCascade = cv2.CascadeClassifier(cascPath)  
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cap = cv2.VideoCapture(0)

def percent(values):
    x = values.item(0)
    y = values.item(1)
    z = values.item(2)
    total = x + y + z
    x = float("{0:.2f}".format(x/total))
    y = float("{0:.2f}".format(y/total))
    z = float("{0:.2f}".format(z/total))
    print("Prob of Class 0 is : ",x,"\nProb of Class 1 is : ",y,"\nProb of Class 2 is : ",z)

while(cap.isOpened()):
    a = datetime.datetime.now()
    ret, image = cap.read()
 
    faces = faceCascade.detectMultiScale(  
      image,  
      scaleFactor=1.05,  
      minNeighbors=5,  
      minSize=(100, 100),  
      flags=cv2.CASCADE_SCALE_IMAGE  
    )  
       
    print("Found {0} faces!".format(len(faces)))  
    for (x, y, w, h) in faces:  
      cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
      dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
      
      landmarks = np.matrix([[p.x, p.y]
      for p in predictor(image, dlib_rect).parts()])  
      
      landmarks_display = landmarks[LEFT_EYE_POINTS + LEFT_EYEBROW_POINTS]  
      for idx, point in enumerate(landmarks_display):
          pos = (point[0, 0], point[0, 1])
      (x, y, w, h) = cv2.boundingRect(landmarks_display)
      roi = image[y:y + h, x:x + w]
      roi = imutils.resize(roi, width=250, height=250, inter=cv2.INTER_CUBIC)

    cv2.imwrite('temp.jpg', roi)
    img = cv2.imread('temp.jpg')
    img = cv2.resize(img,(64,64))
    img = np.reshape(img,[1,64,64,3])
    classes1 = model.predict_classes(img)
    classes2 = model.predict(img)
    class_no = classes1.item(0)
    percent(classes2)

#    print(classes2)
    if (class_no == 0):
        cv2.putText(image,"LEFT", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    elif (class_no == 2):
        cv2.putText(image,"RIGHT", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    else:
        cv2.putText(image,"Middle", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            
    b = datetime.datetime.now()
    print(b-a)
    time.sleep(.5)
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()