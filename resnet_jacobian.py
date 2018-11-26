""" Compute the Jacobians of baseline and cache models
"""
from __future__ import print_function
import os
import sys
import keras
import tensorflow as tf
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Layer
from keras.layers import AveragePooling2D, Input, Flatten, Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from cifar_utils import load_data_cifar10,load_data_cifar100
from keras.utils.data_utils import get_file
import numpy as np
import scipy.io as sio

os.chdir(os.path.dirname(sys.argv[0]))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
job_idx    = int(os.getenv('SLURM_ARRAY_TASK_ID'))

mem_model = True    # whether to use the cache model or the baseline model
cache_only = False  # whether to use the CacheOnly model or the Cache model

saved_models_dir = '/home/eo41/continuous_cache_image_recognition/saved_models/'

TRAINED_WEIGHTS_FILE = saved_models_dir + 'cifar10_ResNet32v1/cifar10_ResNet32v1_model.150.h5'
mem_layers = [82, 92, 103]  # layers to be used for cache memory (ResNet32)
n = 5

# Set optimal hyperparameters for the CacheOnly or the Cache models
if cache_only:
    theta = 72.2222
    lmbd = 1.0
else:
    theta = 85.5556
    lmbd = 0.5111

# Computed depth from supplied model parameter n
depth = n * 6 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, 1)

# Load the CIFAR-10 data.
num_classes = 10
D = load_data_cifar10(flip_train=True, flip_val=True)
data_str = 'cifar10'

x_train = D['X_train']
x_test = D['X_test']
x_val = D['X_val']
y_train = D['Y_train']
y_test = D['Y_test']
y_val = D['Y_val']

# Input image dimensions.
input_shape = x_train.shape[1:]

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# prepare train and test data
x_train = np.concatenate((x_train,x_val),axis=0)
y_train = np.concatenate((y_train,y_val),axis=0)

x_val = x_test
y_val = y_test

# print shapes
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top. v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Function for computing the Jacobian
def jacobian(y, x):
    jacobian_flat = tf.stack( [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y,axis=1)], axis=1)
    return jacobian_flat

# Load model
model = resnet_v1(input_shape=input_shape, depth=depth,  num_classes=num_classes)
print(model_type)
#print(model.layers)

# Load pre-trained weights
model.load_weights(TRAINED_WEIGHTS_FILE)

if mem_model:
    output_list = []
    for i in range(len(mem_layers)):
        output_list.append(model.layers[mem_layers[i]].output)

    # Specify memory
    mem = Model( inputs=model.input, outputs=output_list )

    ### --- Memory keys & values --- ###
    memkeys_list = mem.predict(x_train)
    mem_keys = np.reshape(memkeys_list[0],(x_train.shape[0],-1))
    key_length = mem_keys.shape[1]

    for i in range(len(mem_layers)-1):
        mem_keys = np.concatenate((mem_keys, np.reshape(memkeys_list[i+1],(x_train.shape[0],-1))),axis=1)

    mem_vals = np.float32(y_train)
    ### --- Memory keys & values --- ###

    # Pass items thru memory: note to self -- HANDLE SOME OF THE CONSTANTS BELOW CORRECTLY
    testmem_list = mem(model.input)
    test_mem1 = K.reshape(testmem_list[0],(-1,key_length))
    test_mem2 = K.reshape(testmem_list[1],(-1,key_length))
    test_mem3 = K.reshape(testmem_list[2],(-1,key_length))
    test_mem = K.concatenate((test_mem1, test_mem2, test_mem3),axis=1)

    # Normalize keys and query
    query = test_mem / K.sqrt( K.tile( K.sum(test_mem**2, axis=1, keepdims=True),(1,3*key_length)) )
    key = mem_keys / np.sqrt( np.tile(np.sum(mem_keys**2, axis=1, keepdims=1),(1,mem_keys.shape[1])) )

    # We need to define the keys as placeholders in the graph to get around the memory limit for tf variables.
    place = tf.placeholder(tf.float32, shape=(3*key_length, x_train.shape[0]))
    similarities = K.exp( theta * K.dot(query, place) )
    p_mem = tf.matmul(similarities, mem_vals)
    p_mem = p_mem / K.tile(K.sum(p_mem, axis=1, keepdims=True), (1,num_classes))
    p_combined = (1.0-lmbd) * model.output + lmbd * p_mem

    # Set up the Jacobian
    J = jacobian(p_combined, model.input)
    jacobian_func = K.function([model.input, place, K.learning_phase()], [J])
else:
    J = jacobian(model.output, model.input)
    jacobian_func = K.function([model.input, K.learning_phase()], [J])


# loop over the data to compute jacobian singular values at each training point
bsize = 100
s_val_train_list = []
for i in range(1000):
    if mem_model:
        j_vals = jacobian_func([x_train[(bsize*i):(bsize*(i+1)),:,:,:],key.T,0])[0]
    else:
        j_vals = jacobian_func([x_train[(bsize*i):(bsize*(i+1)),:,:,:],0])[0]
    j_vals = np.reshape(j_vals,(bsize,num_classes,32*32*3))
    s_val_train_list.append(j_vals)

    if (np.remainder(i, 10) == 0):
        print('Training iteration %i'%i)

# loop over the data to compute jacobian singular values at each test point
s_val_test_list = []
for i in range(100):
    if mem_model:
        j_vals = jacobian_func([x_val[(bsize*i):(bsize*(i+1)),:,:,:],key.T,0])[0]
    else:
        j_vals = jacobian_func([x_val[(bsize*i):(bsize*(i+1)),:,:,:],0])[0]
    j_vals = np.reshape(j_vals,(bsize,num_classes,32*32*3))
    s_val_test_list.append(j_vals)

    if (np.remainder(i, 10) == 0):
        print('Test iteration %i'%i)


### --- Compute singular values --- ###
s_val_train_list = np.concatenate(s_val_train_list)
s_val_test_list = np.concatenate(s_val_test_list)

svals_train = np.zeros((100000,num_classes))
svals_test = np.zeros((10000,num_classes))

for i in range(100000):
    m = s_val_train_list[i,:,:]
    s = np.linalg.svd(m,compute_uv=False)
    svals_train[i,:] = s
    if (np.remainder(i, 1000) == 0):
        print('SVD train. iteration %i'%i)

for i in range(10000):
    m = s_val_test_list[i,:,:]
    s = np.linalg.svd(m,compute_uv=False)
    svals_test[i,:] = s
    if (np.remainder(i, 1000) == 0):
        print('SVD test iteration %i'%i)
### --- Compute singular values --- ###


### --- Save results --- ###
sio.savemat('ResNet_%d_jacobian_svals.mat'%depth, {'svals_train': svals_train,
                                                    'svals_test': svals_test})

