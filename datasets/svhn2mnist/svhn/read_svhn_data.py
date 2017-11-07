import cv2 as cv
import scipy.io as sio
import os
import numpy as np
def read_svhn_mat(mat_file):
    data=sio.loadmat(mat_file)
    X = data['X']
    y = data['y']
    y[y==10] = 0 # let 0 denote 0
    return X,y
    
def batch_image_save(output_dir, X, y):
    for idx,label in enumerate(y):
        save_path = os.path.join(output_dir, str(idx)+'.jpg')
        img = X[:,:,:,idx]
        success = cv.imwrite(save_path, img)
    return success

root_dir = './'
#data_home = '/home/data/OCR/SVHN' # server226
data_home = './'
mat_file_list = ['train_32x32.mat', 'test_32x32.mat']
for mat_file in mat_file_list:
    X,y = read_svhn_mat(mat_file)

    output_dir = os.path.join(root_dir, mat_file.split('_')[0] + 'A')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    success = batch_image_save(output_dir, X, y)
    np.savetxt(os.path.join(output_dir,'y.txt'), y, fmt='%d')
    
    
    