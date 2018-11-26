"""Evaluate the ImageNet model
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

theta = 53.0
lmbd = 0.3

n_classes = 1000
train_data_dir = '/beegfs/eo41/processed_train/'
val_data_dir = '/beegfs/eo41/processed_val/'

# Available_layers = [81,84,93,96,103,106,113,116,123,126,133,136,143,146,155,158,165,168,175]
mem_layers = [146]

# Specify model
model = ResNet50(weights='imagenet')
model.summary()

output_list = []
for i in range(len(mem_layers)):
    output_list.append(model.layers[mem_layers[i]].output)

# Specify memory
mem = Model( inputs=model.input, outputs=output_list )

all_mem_keys = []
all_mem_vals = []

for cl_idx in range(n_classes):
    x_train = sio.loadmat(train_data_dir + 'class_%i.mat'%cl_idx)['all_imgs']
    y_train = sio.loadmat(train_data_dir + 'class_%i.mat'%cl_idx)['all_labels']

    # Use the first 280 items of each class in cache
    x_train = np.float32(x_train[:280,:,:,:])
    y_train = np.int32(y_train[0,:280])

    # Memory keys
    memkeys_list = mem.predict(x_train)
    mem_keys = np.reshape(memkeys_list,(x_train.shape[0],-1))

    for i in range(len(mem_layers)-1):
        mem_keys = np.concatenate((mem_keys, np.reshape(memkeys_list[i+1],(x_train.shape[0],-1)) ),axis=1)

    # Memory values
    mem_vals = y_train

    all_mem_keys.append(mem_keys)
    all_mem_vals.append(mem_vals)
    if np.remainder(cl_idx,100)==0:
        print('Iter %i of %i' % (cl_idx, n_classes))

mem_keys = np.concatenate(all_mem_keys,axis=0)
mem_vals = np.concatenate(all_mem_vals,axis=0)
mem_vals = keras.utils.to_categorical(mem_vals, n_classes)

print(mem_keys.shape)
print(mem_vals.shape)

num_batches = 10
val_accs_mem = np.zeros(num_batches)
val_accs_nomem = np.zeros(num_batches)

# Pass validation items thru memory
for val_batch in range(num_batches):
    x_val = sio.loadmat(val_data_dir + 'val_batch_%i'%(val_batch+1))['all_imgs'][:5000,:,:,:]
    y_val = np.loadtxt('ILSVRC2012_validation_ground_truth.txt',usecols=1)[(val_batch*5000):((val_batch+1)*5000)]

    x_val = np.float32(x_val)
    y_val = np.int32(y_val)
    y_val = keras.utils.to_categorical(y_val, n_classes)

    testmem_list = mem.predict(x_val)
    test_mem = np.reshape(testmem_list,(x_val.shape[0],-1))
    for i in range(len(mem_layers)-1):
        test_mem = np.concatenate((test_mem, np.reshape(testmem_list[i+1],(x_val.shape[0],-1))),axis=1)

    # Normalize keys and query
    query = test_mem / np.sqrt( np.tile(np.sum(test_mem**2, axis=1, keepdims=1),(1,test_mem.shape[1])) )
    key = mem_keys / np.sqrt( np.tile(np.sum(mem_keys**2, axis=1, keepdims=1),(1,mem_keys.shape[1])) )

    similarities = np.exp( theta * np.dot(query, key.T) )
    p_mem = np.matmul(similarities, mem_vals)
    p_mem = p_mem / np.repeat(np.sum(p_mem, axis=1, keepdims=True), n_classes, axis=1)

    p_model = model.predict(x_val)

    p_combined = (1.0-lmbd) * p_model + lmbd * p_mem
    pred_combined = np.argmax(p_combined, axis=1)
    y_test_int = np.argmax(y_val, axis=1)
    test_acc = np.mean(pred_combined==y_test_int)
    print('Mem. accuracy:', test_acc)

    # Evaluate trained model without cache
    if val_batch==0:
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])

    scores = model.evaluate(x_val, y_val, verbose=0)
    print('No-mem. accuracy:', scores[1])

    val_accs_mem[val_batch] = test_acc
    val_accs_nomem[val_batch] = scores[1]

sio.savemat('ResNet50_random_jobidx_%i.mat'%job_idx,{'mem_acc': np.mean(val_accs_mem),
                                                     'no_mem_acc': np.mean(val_accs_nomem),
                                                     'theta': theta,
                                                     'lmbd': lmbd})
