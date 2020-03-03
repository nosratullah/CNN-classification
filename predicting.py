import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt

categories = ["Dog", "Cat"]

def prepare(filepath):
    try:
        img_size = 70
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (img_size, img_size))
    except Exception as e:
        pass
    return new_array.reshape(-1, img_size, img_size, 1)

def loadImage(filepath):
    try:
        img_size = 100
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (img_size, img_size))
    except Exception as e:
        pass
    return new_array

model = tf.keras.models.load_model('64x3-CNN.model')
#prediction = model.predict([prepare('PetImages/test/1.jpg')])
#prediction[0][0]

files = os.listdir('PetImages/test')
path = 'PetImages/test'
os.path.join(path, files[2])

plt.figure(figsize=(15,10))
count = 1
for i in os.listdir('PetImages/test'):
    plt.subplot(3,2,count)
    plt.imshow(loadImage(os.path.join(path, i)), 'gray')
    plt.ylabel(categories[int(model.predict([prepare(os.path.join(path, i))])[0][0])])
    plt.xlabel(i)
    '''if (model.predict([prepare(os.path.join(path, i))])[0][0] == 0.0):
        plt.ylabel('Dog')
    else:
        plt.ylabel('Cat')
'''
    count += 1

plt.savefig('prediction.png')
