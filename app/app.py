import keras
from keras.models import load_model
import cv2
import numpy as np




def readImage(FilePath):
    image=cv2.imread(FilePath)
    image=cv2.resize(image,(32,32),interpolation=cv2.INTER_NEAREST)

    return image

def label_names():
    return np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

def predict(image):
    image=image/255
    arr=[]
    arr.append(image)
    arr=np.array(arr)
    image=arr
    model=load_model('./model/model_5.h5')
    prediction=model.predict(image)
    names=label_names()
    return names[np.argmax(prediction)]

image=readImage('./images/test1.png')

prediction=predict(image)

print(prediction)

cv2.imshow('image',image)

cv2.waitKey(0)