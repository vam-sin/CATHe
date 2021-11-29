# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import biovec
import math
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, LeakyReLU, Add
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, matthews_corrcoef, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.utils import resample

# GPU config for Vamsi's Laptop
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 3 * 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

num_classes = 50
# Keras NN Model
def create_model():
    input_ = Input(shape = (1024,))
    x = Dense(1024, activation = "relu")(input_)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation = 'softmax')(x)
    classifier = Model(input_, out)

    return classifier

model = create_model()
print(model.summary())