"""Pre-process the training images
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
from keras.preprocessing import image
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.utils.data_utils import get_file
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import scipy.io as sio

os.chdir(os.path.dirname(sys.argv[0]))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
job_idx    = int(os.getenv('SLURM_ARRAY_TASK_ID'))

model = ResNet50(weights='imagenet')
model.summary()

base_dir = '/beegfs/eo41/ILSVRC2012_img_train/'
proc_train_dir = '/beegfs/eo41/processed_train/'

dir_list = os.listdir(base_dir)
dir_list.sort()
print(dir_list)

dir_itr = 0

for dir_indx in dir_list:
    dir_name = base_dir + dir_indx
    file_list = os.listdir(dir_name)
    file_list.sort()

    all_imgs = []
    for file_indx in file_list:
        file_name = dir_name + '/' + file_indx
        img = image.load_img(file_name, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        all_imgs.append(x)

    all_imgs = np.squeeze(np.asarray(all_imgs))
    all_labels = dir_itr*np.ones(all_imgs.shape[0])
    sio.savemat(proc_train_dir + 'class_%i.mat' % dir_itr, {'all_imgs': all_imgs, 'all_labels': all_labels})
    print('Directory %i of %i'%(dir_itr,len(dir_list)))
    dir_itr = dir_itr + 1

    # some checks to ensure images are properly pre-processed
    preds = model.predict(x)
    print(preds.shape)
    print(np.argmax(preds))
    print(x.shape)

    # decode the results into a list of tuples (class, description, probability)
    print('Predicted:', decode_predictions(preds, top=3)[0])