import cv2 as cv
import scipy.io as sio
import os
import numpy as np
from PIL import Image
def read_svhn_mat(mat_file):
    data=sio.loadmat(mat_file)
    X = data['X']
    y = data['y']
    y[y==10] = 0 # let 0 denote 0
    return X,y


def img_resize(img, img_w, img_h):
    img = Image.fromarray(img).convert('L')
    # print(img.size)
    img = img.resize((img_w, img_h), Image.ANTIALIAS)
    img = np.asarray(img.convert('L'), 'f')
    #print(img.shape)
    return img


def rgb2gray(rgb_img):
    gray_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
    return gray_img


def batch_image_save(output_dir, prefix, X, y, to_rgb2gray=True, to_resize=(28,28), to_save_mat = True):
    '''
    svhn data format is (sample_num, img_w, img_h, channel)
    :param output_dir: output directory
    :param X: numpy array of svhn data X.shape=(sample_num,img_w,img_h,channel)
    :param y: label
    :param to_rgb2gray: default True
    :param to_resize: default (28,28), same as Mnist data shape
    :param to_save_mat: default True
    :return: success, if successfully run the function
    '''
    img_w = to_resize[0]
    img_h = to_resize[1]
    X_mat =  np.zeros([X.shape[-1], img_w, img_h])
    for idx,label in enumerate(y):
        save_path = os.path.join(output_dir, str(idx)+'.jpg')
        img = X[:,:,:,idx]
        print(idx,X.shape)
        if to_rgb2gray:
            img = rgb2gray(img)
        if to_resize:
            img = img_resize(img, img_w, img_h)
            cv.imwrite(save_path, img)
        X_mat[idx] = img

    if to_save_mat:
        file_name = 'svhn_'
        if to_rgb2gray:
            file_name = file_name + 'gray_'
        if to_resize:
            file_name = file_name + prefix + '_' + str(img_w) + 'x' + str(img_h) + '.mat'
        sio.savemat(os.path.join(output_dir, file_name), {'X': X_mat, 'y': y})
    success = 1
    return success


root_dir = './'
data_home = '/home/data/OCR/SVHN' # server226
#data_home = './'
mat_file_list = ['train_32x32.mat', 'test_32x32.mat']
for mat_file in mat_file_list:
    prefix = mat_file.split('_')[0]
    X, y = read_svhn_mat(os.path.join(data_home, mat_file))
    output_dir = os.path.join(root_dir, prefix + 'A')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    success = batch_image_save(output_dir, prefix,  X, y)
    np.savetxt(os.path.join(output_dir,'y.txt'), y, fmt='%d')
