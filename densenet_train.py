"""Trains DenseNet models with different depth on the CIFAR image recognition benchmarks (CIFAR-10 and CIFAR-100).
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

os.chdir(os.path.dirname(sys.argv[0]))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
job_idx    = int(os.getenv('SLURM_ARRAY_TASK_ID'))

VV, DD = np.meshgrid([40,100],[0,1])
VV = VV.flatten()
DD = DD.flatten()

# Training parameters
batch_size = 64
epochs = 300
data_augmentation = True

v = VV[job_idx]         # which version of DenseNet
data_idx = DD[job_idx]  # which dataset: 0 (CIFAR-10) or 1 (CIFAR-100)

# Model name, depth and version
model_type = 'DenseNet_v%d'%v

# Load the CIFAR data.
if data_idx==0:
    num_classes = 10
    D = load_data_cifar10(flip_train=False, flip_val=True)
    data_str = 'cifar10'
else:
    num_classes = 100
    D = load_data_cifar100(flip_train=False, flip_val=True)
    data_str = 'cifar100'

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

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced tenfold after 80 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    if (epoch < 150):
        lr = 0.1
    elif (epoch > 225):
        lr = 0.001
    else:
        lr = 0.01

    print('Learning rate: ', lr)
    return lr

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
    elif blocks == [32, 32, 32]:
        model = Model(inputs, x, name='densenet100')
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

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=lr_schedule(0),momentum=0.9,nesterov=True),
              metrics=['accuracy'])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), '%s_%s_saved_models' % (data_str, model_type))
model_name = '%s_%s_model.{epoch:03d}.h5' % (data_str, model_type)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=0,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.125,
        # randomly shift images vertically
        height_shift_range=0.125,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False)

    # Compute quantities required for featurewise normalization (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train,augment=True,rounds=1)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch= 45000 / batch_size,
                        validation_data=(x_val, y_val),
                        epochs=epochs, verbose=1, workers=1,
                        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])