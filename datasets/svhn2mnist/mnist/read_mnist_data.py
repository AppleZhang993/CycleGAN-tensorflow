import os

import keras
from keras.datasets import mnist

import cv2 as cv
import numpy as np

import scipy.io as sio

(x_train, y_train), (x_test, y_test) = mnist.load_data()

sio.savemat('mnist_train_28x28.mat', {'X':x_train,'y':y_train})
sio.savemat('mnist_test_28x28.mat', {'X':x_test,'y':y_test})

root_dir = './'
#data_home = '/home/data/OCR/SVHN' # server226
data_home = './'

def batch_image_save(output_dir, X, y):
    for idx,label in enumerate(y):
        save_path = os.path.join(output_dir, str(idx)+'.jpg')
        img = X[idx]
        success = cv.imwrite(save_path, img)
    return success
    


output_dir = os.path.join(root_dir, 'trainB')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
success = batch_image_save(output_dir, x_train, y_train)
np.savetxt(os.path.join(output_dir,'y.txt'), y_train, fmt='%d')

output_dir = os.path.join(root_dir, 'testB')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
success = batch_image_save(output_dir, x_test, y_test)
np.savetxt(os.path.join(output_dir,'y.txt'), y_test, fmt='%d')