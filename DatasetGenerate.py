# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:53:39 2018

@author: prant
"""

import cv2
import numpy as np
import dlib  
import imutils
import time


cascPath = 'haarcascade_frontalface_default.xml'  
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

RIGHT_EYE_POINTS = list(range(36, 42))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
  
LEFT_EYE_POINTS = list(range(42, 48))  
LEFT_EYEBROW_POINTS = list(range(22, 27))

faceCascade = cv2.CascadeClassifier(cascPath)
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cap = cv2.VideoCapture(0)

# =============================================================================
# while(cap.isOpened()):
# =============================================================================
for i in range(0,2):
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
   
    cv2.imwrite('Datagen/Right/right_eye'+ str(i) +'.jpg', roi1)
    cv2.imwrite('Datagen/Left/left_eye'+ str(i) +'.jpg', roi2)

    time.sleep(.25)
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()