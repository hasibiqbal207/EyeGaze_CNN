# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 23:18:40 2018

@author: prant
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(Conv2D(24, (7, 7), padding='same', activation='relu', input_shape = (64, 64, 3)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(24, (5, 5), padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(24, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(activation='softmax', units = 3))

classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset_eyebrow/left_eye/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset_eyebrow/left_eye/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


classifier.fit_generator(training_set,
                         samples_per_epoch = 383,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 39)

classifier.save('lefteyemodel.h5')