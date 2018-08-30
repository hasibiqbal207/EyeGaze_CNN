# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 23:18:53 2018

@author: prant
"""

from keras.models import load_model
import cv2
import numpy as np

model = load_model('model.h5')
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

img = cv2.imread('r1.jpg')
img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,3])
classes = model.predict_classes(img)
print(classes)

img = cv2.imread('m1.jpg')
img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,3])
classes = model.predict_classes(img)
print(classes)

img = cv2.imread('l1.jpg')
img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,3])
classes = model.predict_classes(img)
print(classes)