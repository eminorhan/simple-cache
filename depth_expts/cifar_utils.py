# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:01:29 2016 @author: emin
"""
import numpy as np

IMAGE_SIZE = 32
NUM_CHANNELS = 3
cifar_dir = '/home/eo41/understanding_resnets/experiments/data_utils/'

# ################## Prepare the CIFAR dataset ##################
def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def load_data_cifar10(flip_train=False, flip_val=False):
    xs = []
    ys = []
    for j in range(5):
      d = unpickle(cifar_dir+'cifar-10-batches-py/data_batch_%i'%(j+1))
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle(cifar_dir+'cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs) / np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)) #.transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:60000],axis=0)
    pixel_std  = np.std(x[0:60000],axis=0)
    x = (x - pixel_mean) / pixel_std

    # create mirrored training images
    X_train = x[0:45000,:,:,:]
    Y_train = y[0:45000]
    if flip_train:
        X_train_flip = X_train[:,:,::-1,:]
        Y_train_flip = Y_train
        X_train = np.concatenate((X_train,X_train_flip),axis=0) # add flipped version
        Y_train = np.concatenate((Y_train,Y_train_flip),axis=0) # add flipped version

    # create mirrored validation images
    X_val = x[45000:50000,:,:,:]
    Y_val = y[45000:50000]
    if flip_val:
        X_val_flip = X_val[:,:,::-1,:]
        Y_val_flip = Y_val
        X_val = np.concatenate((X_val, X_val_flip), axis=0) # add flipped version
        Y_val = np.concatenate((Y_val, Y_val_flip), axis=0) # add flipped version

    # test images
    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict( X_train= X_train.astype('float32'),
                 Y_train= Y_train.astype('int32'),
                 X_test = X_test.astype('float32'),
                 Y_test = Y_test.astype('int32'),
                 X_val=X_val.astype('float32'),
                 Y_val=Y_val.astype('int32')
                 )

def load_data_cifar100(flip_train=False, flip_val=False):
    xs = []
    ys = []

    d = unpickle(cifar_dir+'cifar-100-python/train')
    x = d['data']
    y = d['fine_labels']
    xs.append(x)
    ys.append(y)

    d = unpickle(cifar_dir+'cifar-100-python/test')
    xs.append(d['data'])
    ys.append(d['fine_labels'])

    x = np.concatenate(xs) / np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)) #.transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:60000],axis=0)
    pixel_std  = np.std(x[0:60000],axis=0)
    x = (x - pixel_mean) / pixel_std

    # create mirrored training images
    X_train = x[0:45000,:,:,:]
    Y_train = y[0:45000]
    if flip_train:
        X_train_flip = X_train[:,:,::-1,:]
        Y_train_flip = Y_train
        X_train = np.concatenate((X_train,X_train_flip),axis=0) # add flipped version
        Y_train = np.concatenate((Y_train,Y_train_flip),axis=0) # add flipped version

    # create mirrored validation images
    X_val = x[45000:50000,:,:,:]
    Y_val = y[45000:50000]
    if flip_val:
        X_val_flip = X_val[:,:,::-1,:]
        Y_val_flip = Y_val
        X_val = np.concatenate((X_val, X_val_flip), axis=0) # add flipped version
        Y_val = np.concatenate((Y_val, Y_val_flip), axis=0) # add flipped version

    # test images
    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict( X_train= X_train.astype('float32'),
                 Y_train= Y_train.astype('int32'),
                 X_test = X_test.astype('float32'),
                 Y_test = Y_test.astype('int32'),
                 X_val=X_val.astype('float32'),
                 Y_val=Y_val.astype('int32')
                 )
