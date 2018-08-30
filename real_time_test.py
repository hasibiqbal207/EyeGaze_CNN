# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 22:58:41 2018

@author: prant
"""

from keras.models import load_model
import cv2
import numpy as np
import dlib  
import imutils
import time
import datetime

model_left = load_model('lefteyemodel.h5')
model_left.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

model_right = load_model('righteyemodel.h5')
model_right.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

cascPath = 'haarcascade_frontalface_default.xml'  
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

RIGHT_EYE_POINTS = list(range(36, 42))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
  
LEFT_EYE_POINTS = list(range(42, 48))  
LEFT_EYEBROW_POINTS = list(range(22, 27))

faceCascade = cv2.CascadeClassifier(cascPath)  
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cap = cv2.VideoCapture(0)

def percent(values_left, values_right):
    x_l = values_left.item(0)
    y_l = values_left.item(1)
    z_l = values_left.item(2)
    
    x_r = values_right.item(0)
    y_r = values_right.item(1)
    z_r = values_right.item(2)
    
    total_l = x_l + y_l + z_l
    total_r = x_r + y_r + z_r
    
    x_l = float("{0:.2f}".format(x_l/total_l))
    y_l = float("{0:.2f}".format(y_l/total_l))
    z_l = float("{0:.2f}".format(z_l/total_l))
    
    x_r = float("{0:.2f}".format(x_r/total_r))
    y_r = float("{0:.2f}".format(y_r/total_r))
    z_r = float("{0:.2f}".format(z_r/total_r))
    
    x = (x_l + x_r)/2
    y = (y_l + y_r)/2
    z = (z_l + z_r)/2    
    print("Prob of Class 0 is : ",x,"\nProb of Class 1 is : ",y,"\nProb of Class 2 is : ",z)
    
    if (x >= 0.80):
        cv2.putText(image,"LEFT", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    elif (z >= 0.80):
        cv2.putText(image,"RIGHT", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    else:
        cv2.putText(image,"Middle", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

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
      # Converting the OpenCV rectangle coordinates to Dlib rectangle  
      dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
      
      landmarks = np.matrix([[p.x, p.y]
      for p in predictor(image, dlib_rect).parts()])  
      
      landmarks_display_right = landmarks[RIGHT_EYE_POINTS + RIGHT_EYEBROW_POINTS]  
      for idx, point in enumerate(landmarks_display_right):
          pos = (point[0, 0], point[0, 1])
 #         cv2.circle(image, pos, 1, color=(0, 0, 255), thickness=-1)
      (x, y, w, h) = cv2.boundingRect(landmarks_display_right)
      roi1 = image[y:y + h, x:x + w]
      roi1 = imutils.resize(roi1, width=250, height=250, inter=cv2.INTER_CUBIC)
      
      landmarks_display_left = landmarks[LEFT_EYE_POINTS + LEFT_EYEBROW_POINTS]  
      for idx, point in enumerate(landmarks_display_left):
          pos = (point[0, 0], point[0, 1])
 #         cv2.circle(image, pos, 1, color=(0, 0, 255), thickness=-1)
      (x, y, w, h) = cv2.boundingRect(landmarks_display_left)
      roi2 = image[y:y + h, x:x + w]
      roi2 = imutils.resize(roi2, width=250, height=250, inter=cv2.INTER_CUBIC)
      
    cv2.imwrite('temp_right.jpg', roi1)
    img = cv2.imread('temp_right.jpg')
    img = cv2.resize(img,(64,64))
    img = np.reshape(img,[1,64,64,3])
    classes_r81 = model_right.predict_classes(img)
    classes_r82 = model_right.predict(img)
    class_no = classes_r81.item(0)

    cv2.imwrite('temp_left.jpg', roi2)
    img = cv2.imread('temp_left.jpg')
    img = cv2.resize(img,(64,64))
    img = np.reshape(img,[1,64,64,3])
    classes_l81 = model_left.predict_classes(img)
    classes_l82 = model_left.predict(img)
    class_no = classes_l81.item(0)
    
    percent(classes_l82, classes_r82)
            
    b = datetime.datetime.now()
    print(b-a)
    time.sleep(.5)
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


