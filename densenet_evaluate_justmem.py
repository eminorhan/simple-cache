"""Evaluate DenseNet CacheOnly models on CIFAR benchmarks.
"""

from __future__ import print_function
import os
import sys
import keras
from keras import regularizers
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Concatenate, Dropout
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from cifar_utils import load_data_cifar10, load_data_cifar100
from imagenet_utils import _obtain_input_shape
import numpy as np
import scipy.io as sio

os.chdir(os.path.dirname(sys.argv[0]))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
job_idx    = int(os.getenv('SLURM_ARRAY_TASK_ID'))

data_idx = 0                # 0 for CIFAR-10, something else for CIFAR-100
model_name = 'densenet40'   # 'densenet40' or 'densenet100'

saved_models_dir = '/home/eo41/continuous_cache_image_recognition/saved_models/'

# ---------------------------------------------------------------------------
# Load the CIFAR data.
if data_idx==0:
    num_classes = 10
    D = load_data_cifar10(flip_train=True, flip_val=True)
    data_str = 'cifar10'

    if model_name == 'densenet40':
        TRAINED_WEIGHTS_FILE = saved_models_dir + 'cifar10_DenseNet_v40/cifar10_DenseNet_v40_model.253.h5'
        mem_layers = [136, 152]
        v = 40
        # Setup hyperparam theta to optimal value
        theta = 61.1111
        lmbd = 1.0
    elif model_name == 'densenet100':
        TRAINED_WEIGHTS_FILE = saved_models_dir + 'cifar10_DenseNet_v100/cifar10_DenseNet_v100_model.280.h5'
        mem_layers = [388]
        v = 100
        # Setup hyperparam theta to optimal value
        theta = 61.1111
        lmbd = 1.0
else:
    num_classes = 100
    D = load_data_cifar100(flip_train=True, flip_val=True)
    data_str = 'cifar100'

    if model_name == 'densenet40':
        TRAINED_WEIGHTS_FILE = saved_models_dir + 'cifar100_DenseNet_v40/cifar100_DenseNet_v40_model.153.h5'
        v = 40
        mem_layers = [136, 152]
        # Setup hyperparam theta to optimal value
        theta = 61.1111
        lmbd = 1.0
    elif model_name == 'densenet100':
        TRAINED_WEIGHTS_FILE = saved_models_dir + 'cifar100_DenseNet_v100/cifar100_DenseNet_v100_model.250.h5'
        mem_layers = [388]
        v = 100
        # Setup hyperparam theta to optimal value
        theta = 61.1111
        lmbd = 1.0

model_type = 'DenseNet_v%d'%v

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

print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)

def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 12, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=True, name=name + '_conv', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=True, name=name + '_2_conv', kernel_regularizer=regularizers.l2(0.0001))(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks,
             include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=10):
    """Instantiates the DenseNet architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with
    TensorFlow, Theano, and CNTK. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(img_input)
    x = Conv2D(16, 3, strides=1, use_bias=True, name='conv1/conv', kernel_regularizer=regularizers.l2(0.0001))(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 1, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 1, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000', kernel_regularizer=regularizers.l2(0.0001))(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, x, name='densenet201')
    elif blocks == [12, 12, 12]:
        model = Model(inputs, x, name='densenet40')
    else:
        model = Model(inputs, x, name='densenet')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if blocks == [6, 12, 24, 16]:
                weights_path = get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET121_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='0962ca643bae20f9b6771cb844dca3b0')
            elif blocks == [6, 12, 32, 32]:
                weights_path = get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET169_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='bcf9965cf5064a5f9eb6d7dc69386f43')
            elif blocks == [6, 12, 48, 32]:
                weights_path = get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET201_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='7bb75edd58cb43163be7e0005fbe95ef')
        else:
            if blocks == [6, 12, 24, 16]:
                weights_path = get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET121_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='4912a53fbd2a69346e7f2c0b5ec8c6d3')
            elif blocks == [6, 12, 32, 32]:
                weights_path = get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET169_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='50662582284e4cf834ce40ab4dfa58c6')
            elif blocks == [6, 12, 48, 32]:
                weights_path = get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET201_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='1c2de60ee40562448dbac34a0737e798')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def DenseNet40(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=10):
    return DenseNet([12, 12, 12],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


def DenseNet100(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=10):
    return DenseNet([32, 32, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


def DenseNet121(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=10):
    return DenseNet([6, 12, 24, 16],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


def DenseNet169(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=10):
    return DenseNet([6, 12, 32, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


def DenseNet201(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=10):
    return DenseNet([6, 12, 48, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


if (v==40):
    model = DenseNet40(input_shape=input_shape, classes=num_classes)
elif (v==100):
    model = DenseNet100(input_shape=input_shape, classes=num_classes)

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

# Score trained model.
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1,momentum=0.9,nesterov=True), metrics=['accuracy']) # this is a dummy compile
scores = model.evaluate(x_val, y_val, verbose=0)
print('No-mem. accuracy:', scores[1])

sio.savemat('DenseNet_v%d_justmem_evaluate.mat' %v, {'mem_acc': test_acc,
                                                     'no_mem_acc': scores[1],
                                                     'theta': theta,
                                                     'lmbd': lmbd})