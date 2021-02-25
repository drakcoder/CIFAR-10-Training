import numpy as np
import cv2
from PIL import Image  

def unpickle(file):
    with open(file,'rb') as fo:
        dicts=pickle.load(fo,encoding='bytes')

    return dicts

def load_data(batchNumber):
    data=np.load('./unpickled_data/data_'+str(batchNumber)+'.npy')
    labels=np.load('./unpickled_data/labels_'+str(batchNumber)+'.npy')

    data=data.reshape((data.shape[0]),3,32,32).transpose(0,2,3,1)

    return data,labels

def label_names():
    return np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])


features=[]
labels=[]
for i in range(1,6):
    f,l=load_data(i)
    features.extend(f)
    labels.extend(l)

features=np.array(features)
labels=np.array(labels)
labelNames=label_names()

exampleX=np.load('./unpickled_data/X_test.npy')
exampleY=np.load('./unpickled_data/Y_test.npy')

exampleX=exampleX.reshape(10000,3,32,32).transpose(0,2,3,1)

cv2.imshow('img',exampleX[7])

print(exampleY[7])

cv2.waitKey(0)
