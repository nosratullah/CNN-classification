import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model,save_model
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import pickle
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time


X = pickle.load(open("PetImages/X.pickle",'rb'))
y = pickle.load(open("PetImages/Y.pickle",'rb'))

X = X/255.0
y = np.array(y)

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            Name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorBoard = TensorBoard(log_dir='logs/{}'.format(Name))
            print(Name)
            model = Sequential()
            model.add(Conv2D(64, (3,3) ,input_shape = X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten()) # Convert 3d feature map to 1D vector
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

            model.fit(X, y, batch_size=32, epochs=10 ,validation_split=0.3, callbacks=[tensorBoard])

#model.save('{}.model'.format(Name))
model.save('64x3-CNN.model')
