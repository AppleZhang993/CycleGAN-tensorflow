from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import h5py
import scipy.io as sio
import numpy as np
import argparse
import os
import tensorflow as tf
tf.set_random_seed(19)
from model import cyclegan

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='svhn2mnist', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=32, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=28, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')

args = parser.parse_args()


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

def read_mat(mat_file):
    data=sio.loadmat(mat_file)
    X = data['X'] #(32,32,3,#)
    y = data['y']
    return X, y


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)
print(set(y_test))


#data_home = '../datasets/svhn2mnist/'
#x_train, y_train = read_mat(os.path.join(data_home, 'svhn/svhn_gray_train_28x28.mat'))
#x_test, y_test = read_mat(os.path.join(data_home, 'svhn/svhn_gray_test_28x28.mat'))


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('y_train shape:', y_train.shape,'y_test.shape',y_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])



model.load_weights('./classifier/mnist_weights.h5')
score = model.evaluate(x_test, y_test.reshape(-1,), verbose=0)
print('Mnist Test loss:', score[0])
print('Mnist Test accuracy:', score[1])

data_home = './datasets/svhn2mnist/'
x_train, y_train = read_mat(os.path.join(data_home, 'svhn/svhn_gray_train_28x28.mat'))
x_test, y_test = read_mat(os.path.join(data_home, 'svhn/svhn_gray_test_28x28.mat'))

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

score = model.evaluate(x_train, y_train.reshape(-1,), verbose=0)
print('svhn Train loss:', score[0])
print('svhn Train accuracy:', score[1])
score = model.evaluate(x_test, y_test, verbose=0)
print('svhn Test loss:', score[0])
print('svhn Test accuracy:', score[1])

print("begining cycle gan.....")

with tf.Session(config=tfconfig) as sess:
    domain_adapation_model = cyclegan(sess, args)
    x_test_c = np.zeros(x_test.shape)
    batch_size = 2
    batch_num = int(x_test.shape[0]/batch_size)
    for idx in range(batch_num):
        print(idx)
        x_test_c[idx*batch_size:(idx+1)*batch_size] = domain_adapation_model.pix2pix_cylce_gan(args, x_test[idx*batch_size:(idx+1)*batch_size])
    score = model.evaluate(x_test_c, y_test, verbose=0)
    print('svhn Test loss after cycle gan:', score[0])
    print('svhn Test accuracy after cycle gan:', score[1])
    #x_train_c = domain_adapation_model.pix2pix_cylce_gan(args, x_train)
    #score = model.evaluate(x_train_c, y_train, verbose=0)
    #print('svhn Train loss after cycle gan:', score[0])
    #print('svhn Train accuracy after cycle gan:', score[1])
