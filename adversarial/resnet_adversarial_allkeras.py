""" Run white-box adversarial attacks on the baseline and cache models
"""
from __future__ import print_function
import os
import sys
import keras
import tensorflow as tf
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Layer
from keras.layers import AveragePooling2D, Input, Flatten, Lambda, Reshape, Concatenate, RepeatVector, Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from cifar_utils import load_data_cifar10
from keras.utils.data_utils import get_file
import numpy as np
import scipy.io as sio
import foolbox
from MultiInputKerasModel import MultiInputKerasModel
from TwoInputKerasModel import TwoInputKerasModel

os.chdir(os.path.dirname(sys.argv[0]))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
job_idx    = int(os.getenv('SLURM_ARRAY_TASK_ID'))
np.random.seed(job_idx)

num_classes = 10
mem_model = True      # whether to use the cache model or the baseline model
cache_only = True     # whether to use the CacheOnly model
attack_name = 'fgsm'  # attack: 'fgsm', 'ifgsm', 'sp', 'gb'

saved_models_dir = '/home/eo41/continuous_cache_image_recognition/saved_models/'

TRAINED_WEIGHTS_FILE = saved_models_dir + 'cifar10_ResNet32v1/cifar10_ResNet32v1_model.150.h5'
mem_layers = [82,92,103]  # layers to be used for cache memory (ResNet32)
n = 5

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

# Load the CIFAR10 data.
D = load_data_cifar10(flip_train=True,flip_val=True)
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
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Function for computing the Jacobian
def jacobian(y, x):
    jacobian_flat = tf.stack( [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y,axis=1)], axis=1)
    return jacobian_flat

def merged_div(concat_vec):
    return concat_vec[0]/concat_vec[1]

def merged_dot(concat_vec):
    return K.dot(concat_vec[0],concat_vec[1])

# Load model
model = resnet_v1(input_shape=input_shape, depth=depth)
print(model_type)
model.summary()

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

    key = mem_keys / np.sqrt( np.tile(np.sum(mem_keys**2, axis=1, keepdims=1),(1,mem_keys.shape[1])) )
    mem_vals = np.float32(y_train)

    ### --- Memory keys & values --- ###
    # Pass items thru memory
    testmem_list = mem(model.input)
    test_mem1 = Reshape((key_length,))(testmem_list[0])
    test_mem2 = Reshape((key_length,))(testmem_list[1])
    test_mem3 = Reshape((key_length,))(testmem_list[2])
    test_mem = Concatenate(axis=1)([test_mem1, test_mem2, test_mem3])

    # Normalize keys and query
    test_mem_norm = Lambda(lambda X: X**2)(test_mem)
    test_mem_norm = Lambda(lambda X: K.sum(X, axis=1, keepdims=True))(test_mem_norm)
    test_mem_norm = Lambda(lambda X: K.tile( X, (1,3*key_length)) )(test_mem_norm)
    test_mem_norm = Lambda(lambda X: K.sqrt(X))(test_mem_norm)
    query = Lambda(merged_div)([test_mem, test_mem_norm])

    # We need to define the keys as placeholders in the graph to get around the memory limit for tf variables.
    place = Input(batch_shape=(3*key_length,x_train.shape[0]))
    qp_dot = Lambda(merged_dot)([query, place])
    similarities = Lambda(lambda X: K.exp(theta * X))(qp_dot)
    p_mem = Lambda(lambda X: tf.matmul(X, mem_vals))(similarities)

    p_mem_norm = Lambda(lambda X: K.sum(X, axis=1, keepdims=True))(p_mem)
    p_mem_norm = Lambda(lambda X: K.tile( X, (1,10)) )(p_mem_norm)
    p_mem = Lambda(merged_div)([p_mem, p_mem_norm])

    p_combined = Add()([Lambda(lambda X: (1.0-lmbd)*X)(model.output), Lambda(lambda X: lmbd*X)(p_mem)])

    keras_model = Model(inputs=[model.input,place],outputs=p_combined)
    foolbox_model = MultiInputKerasModel(keras_model, (x_val.min(), x_val.max()), key.T, 0)
else:
    keras_model = model
    foolbox_model = TwoInputKerasModel(keras_model, (x_val.min(), x_val.max()), 0)

criterion = foolbox.criteria.Misclassification()

if attack_name=='fgsm':
    attack = foolbox.attacks.FGSM(foolbox_model, criterion)
elif attack_name=='ifgsm':
    attack = foolbox.attacks.IterativeGradientSignAttack(foolbox_model, criterion)
elif attack_name=='sp':
    attack = foolbox.attacks.SinglePixelAttack(foolbox_model, criterion)
elif attack_name=='gb':
    attack = foolbox.attacks.GaussianBlurAttack(foolbox_model, criterion)

num_adv = 2500
adv_indices = np.arange(x_val.shape[0])
np.random.shuffle(adv_indices)
example_images = x_val[adv_indices[:num_adv], :, :, :]
example_labels = y_val[adv_indices[:num_adv], :]
relative_perturbation_norms = np.zeros(num_adv)

real_imgs = []
adv_imgs = []
successful_labels = []

for img_indx in range(example_images.shape[0]):

    real_image = example_images[img_indx, :, : , :]
    model_prediction = foolbox_model.predictions(real_image)
    label = np.argmax(model_prediction)

    if attack_name=='fgsm':
        adversarial_image = attack(real_image, label=label, epsilons=50, max_epsilon=0.5)
    elif attack_name=='ifgsm':
        adversarial_image = attack(real_image, label=label, epsilons=50, steps=10)
    elif attack_name=='sp':
        adversarial_image = attack(real_image, label=label, max_pixels=1000) # SinglePixelAttack
    elif attack_name=='gb':
        adversarial_image = attack(real_image, label=label, epsilons=50) # Gaussian blur

    # print(img_indx)

    if adversarial_image is not None:
        relative_perturbation_norms[img_indx] = np.linalg.norm(adversarial_image - real_image) / np.linalg.norm(real_image)
        print(relative_perturbation_norms[img_indx])

        if np.remainder(img_indx,1)==0:
            real_imgs.append(real_image)
            adv_imgs.append(adversarial_image)
            successful_labels.append(example_labels[img_indx, :])

### --- Save results --- ###
real_imgs = np.asarray(real_imgs)
adv_imgs = np.asarray(adv_imgs)
successful_labels = np.asarray(successful_labels)
print(adv_imgs.shape)
print(successful_labels.shape)
sio.savemat('ResNet_%d_adversarials.mat'%depth, {'relative_perturbation_norms': relative_perturbation_norms,
                                                 'real_imgs': real_imgs,
                                                 'adv_imgs': adv_imgs,
                                                 'example_labels': example_labels,
                                                 'successful_labels': successful_labels
                                                })

