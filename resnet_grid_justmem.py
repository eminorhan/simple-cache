"""Hyperparameter search over theta for the ResNet CacheOnly models.
"""
from __future__ import print_function
import os
import sys
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from cifar_utils import load_data_cifar10, load_data_cifar100
from keras.utils.data_utils import get_file
import numpy as np
import scipy.io as sio

os.chdir(os.path.dirname(sys.argv[0]))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
job_idx    = int(os.getenv('SLURM_ARRAY_TASK_ID'))

data_idx = 0             # data index: 0 (CIFAR-10) or 1 (CIFAR-100)
model_name = 'resnet32'  # model name: 'resnet20', 'resnet32', or 'resnet56'

saved_models_dir = '/home/eo41/continuous_cache_image_recognition/saved_models/'

# ---------------------------------------------------------------------------
# Setup hyperparams: theta
theta_range = np.linspace(10.0,90.0,33)

theta = theta_range[job_idx]
lmbd = 1.0

# ---------------------------------------------------------------------------
# Load the CIFAR data and model.
if data_idx==0:
    num_classes = 10
    D = load_data_cifar10(flip_train=True, flip_val=True)
    data_str = 'cifar10'

    if model_name == 'resnet20':
        TRAINED_WEIGHTS_FILE = saved_models_dir + 'cifar10_ResNet20v1/cifar10_ResNet20v1_model.178.h5'
        mem_layers = [49, 57, 64]  # layers to be used for cache memory (ResNet20)
        n = 3
    elif model_name == 'resnet32':
        TRAINED_WEIGHTS_FILE = saved_models_dir + 'cifar10_ResNet32v1/cifar10_ResNet32v1_model.150.h5'
        mem_layers = [82, 92, 103]  # layers to be used for cache memory (ResNet32)
        n = 5
    elif model_name == 'resnet56':
        TRAINED_WEIGHTS_FILE = saved_models_dir + 'cifar10_ResNet56v1/cifar10_ResNet56v1_model.167.h5'
        mem_layers = [131, 160, 188]  # layers to be used for cache memory (ResNet56)
        n = 9
else:
    num_classes = 100
    D = load_data_cifar100(flip_train=True, flip_val=True)
    data_str = 'cifar100'

    if model_name == 'resnet20':
        TRAINED_WEIGHTS_FILE = saved_models_dir + 'cifar100_ResNet20v1/cifar100_ResNet20v1_model.158.h5'
        mem_layers = [49, 57, 64]  # layers to be used for cache memory (ResNet20)
        n = 3
    elif model_name == 'resnet32':
        TRAINED_WEIGHTS_FILE = saved_models_dir + 'cifar100_ResNet32v1/cifar100_ResNet32v1_model.143.h5'
        mem_layers = [82, 92, 103]  # layers to be used for cache memory (ResNet32)
        n = 5
    elif model_name == 'resnet56':
        TRAINED_WEIGHTS_FILE = saved_models_dir + 'cifar100_ResNet56v1/cifar100_ResNet56v1_model.156.h5'
        mem_layers = [131, 160, 188]  # layers to be used for cache memory (ResNet56)
        n = 9

# Computed depth from supplied model parameter n
depth = n * 6 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, 1)

x_train = D['X_train']
x_test = D['X_test']
x_val = D['X_val']
y_train = D['Y_train']
y_test = D['Y_test']
y_val = D['Y_val']

# Input image dimensions.
input_shape = x_train.shape[1:]

print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

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
                # linear projection residual shortcut connection to match
                # changed dims
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

# Load model
model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)

#model.summary()
print(model_type)

# Load pre-trained weights
model.load_weights(TRAINED_WEIGHTS_FILE)

output_list = []
for i in range(len(mem_layers)):
    output_list.append(model.layers[mem_layers[i]].output)

# Specify memory
mem = Model( inputs=model.input, outputs=output_list )

# Memory keys
memkeys_list = mem.predict(x_train)
mem_keys = np.reshape(memkeys_list[0],(x_train.shape[0],-1))
for i in range(len(mem_layers)-1):
    mem_keys = np.concatenate((mem_keys, np.reshape(memkeys_list[i+1],(x_train.shape[0],-1))),axis=1)

# Memory values
mem_vals = y_train

# Pass items thru memory
testmem_list = mem.predict(x_val)
test_mem = np.reshape(testmem_list[0],(x_val.shape[0],-1))
for i in range(len(mem_layers)-1):
    test_mem = np.concatenate((test_mem, np.reshape(testmem_list[i+1],(x_val.shape[0],-1))),axis=1)

# Normalize keys and query
query = test_mem / np.sqrt( np.tile(np.sum(test_mem**2, axis=1, keepdims=1),(1,test_mem.shape[1])) )
key = mem_keys / np.sqrt( np.tile(np.sum(mem_keys**2, axis=1, keepdims=1),(1,mem_keys.shape[1])) )

similarities = np.exp( theta * np.dot(query, key.T) )
p_mem = np.matmul(similarities, mem_vals)
p_mem = p_mem / np.repeat(np.sum(p_mem, axis=1, keepdims=True), num_classes, axis=1)

p_model = model.predict(x_val)

p_combined = (1.0-lmbd) * p_model + lmbd * p_mem
pred_combined = np.argmax(p_combined, axis=1)
y_test_int = np.argmax(y_val, axis=1)
test_acc = np.mean(pred_combined==y_test_int)

print('Mem. shape:', mem_keys.shape)
print('Mem. accuracy:', test_acc)

# Evaluate trained model without cache
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])
scores = model.evaluate(x_val, y_val, verbose=0)
print('No-mem. accuracy:', scores[1])

sio.savemat('ResNet%d_justmem_jobidx%i.mat' % (depth,job_idx), {'mem_acc': test_acc,
                                                        'no_mem_acc': scores[1],
                                                        'theta': theta,
                                                        'lmbd': lmbd})