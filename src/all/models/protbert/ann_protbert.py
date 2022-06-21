'''
ANN model trained on the PB embeddings
'''

# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
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
import tensorflow_addons as tfa
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

# dataset import
# train 
ds_train = pd.read_csv('Y_Train_SF.csv')
y_train = list(ds_train["SF"])

filename = 'SF_Train_ProtBert.npz'
X_train = np.load(filename)['arr_0']
filename = 'Other_Train_US.npz'
X_train_other = np.load(filename)['arr_0']

X_train = np.concatenate((X_train, X_train_other), axis=0)

for i in range(len(X_train_other)):
    y_train.append('other')

# val
ds_val = pd.read_csv('Y_Val_SF.csv')
y_val = list(ds_val["SF"])

filename = 'SF_Val_ProtBert.npz'
X_val = np.load(filename)['arr_0']

filename = 'Other_Val_US.npz'
X_val_other = np.load(filename)['arr_0']

X_val = np.concatenate((X_val, X_val_other), axis=0)

for i in range(len(X_val_other)):
    y_val.append('other')

# test
ds_test = pd.read_csv('Y_Test_SF.csv')
y_test = list(ds_test["SF"])

filename = 'SF_Test_ProtBert.npz'
X_test = np.load(filename)['arr_0']

filename = 'Other_Test_US.npz'
X_test_other = np.load(filename)['arr_0']

X_test = np.concatenate((X_test, X_test_other), axis=0)

for i in range(len(X_test_other)):
    y_test.append('other')

# y process
y_tot = []

for i in range(len(y_train)):
    y_tot.append(y_train[i])

for i in range(len(y_val)):
    y_tot.append(y_val[i])

for i in range(len(y_test)):
    y_tot.append(y_test[i])

le = preprocessing.LabelEncoder()
le.fit(y_tot)

y_train = np.asarray(le.transform(y_train))
y_val = np.asarray(le.transform(y_val))
y_test = np.asarray(le.transform(y_test))

num_classes = len(np.unique(y_tot))
print(num_classes)
print("Loaded X and y")

X_train, y_train = shuffle(X_train, y_train, random_state=42)
print("Shuffled")

# generator
def bm_generator(X_t, y_t, batch_size):
    val = 0

    while True:
        X_batch = []
        y_batch = []

        for j in range(batch_size):

            if val == len(X_t):
                val = 0

            X_batch.append(X_t[val])
            y_enc = np.zeros((num_classes))
            y_enc[y_t[val]] = 1
            y_batch.append(y_enc)
            val += 1

        X_batch = np.asarray(X_batch)
        y_batch = np.asarray(y_batch)

        yield X_batch, y_batch

# batch size
bs = 256

# Keras NN Model
def create_model():
    input_ = Input(shape = (1024,))
    x = Dense(1024, activation = "relu", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(input_)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(1024, activation = "relu", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x) 
    out = Dense(num_classes, activation = 'softmax', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
    classifier = Model(input_, out)

    return classifier

# training
num_epochs = 200

with tf.device('/gpu:0'):
    # model
    model = create_model()

    # adam optimizer
    opt = keras.optimizers.Adam(learning_rate = 1e-5)
    model.compile(optimizer = "adam", loss = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO), metrics=['accuracy'])

    # callbacks
    mcp_save = keras.callbacks.ModelCheckpoint('saved_models/ann_pb.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=20, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)
    callbacks_list = [reduce_lr, mcp_save, early_stop]

    # test and train generators
    train_gen = bm_generator(X_train, y_train, bs)
    val_gen = bm_generator(X_val, y_val, bs)
    test_gen = bm_generator(X_test, y_test, bs)
    history = model.fit_generator(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(bs)), verbose=1, validation_data = val_gen, validation_steps = len(X_val)/bs, workers = 0, shuffle = True, callbacks = callbacks_list)
    model = load_model('saved_models/ann_pb.h5')

    print("Validation")
    y_pred_val = model.predict(X_val)
    f1_score_val = f1_score(y_val, y_pred_val.argmax(axis=1), average = 'weighted')
    acc_score_val = accuracy_score(y_val, y_pred_val.argmax(axis=1))
    print("F1 Score: ", f1_score_val)
    print("Acc Score", acc_score_val)

    print("Regular Testing")
    y_pred_test = model.predict(X_test)
    f1_score_test = f1_score(y_test, y_pred_test.argmax(axis=1), average = 'macro')
    acc_score_test = accuracy_score(y_test, y_pred_test.argmax(axis=1))
    mcc_score = matthews_corrcoef(y_test, y_pred_test.argmax(axis=1))
    bal_acc = balanced_accuracy_score(y_test, y_pred_test.argmax(axis=1))
    print("F1 Score: ", f1_score_test)
    print("Acc Score: ", acc_score_test)
    print("MCC: ", mcc_score)
    print("Bal Acc: ", bal_acc)

    print("Bootstrapping Results")
    num_iter = 1000
    f1_arr = []
    acc_arr = []
    mcc_arr = []
    bal_arr = []
    for it in range(num_iter):
        # print("Iteration: ", it)
        X_test_re, y_test_re = resample(X_test, y_test, n_samples = len(y_test), random_state=it)
        y_pred_test_re = model.predict(X_test_re)
        #print(y_test_re)
        f1_arr.append(f1_score(y_test_re, y_pred_test_re.argmax(axis=1), average = 'macro'))
        acc_arr.append(accuracy_score(y_test_re, y_pred_test_re.argmax(axis=1)))
        mcc_arr.append(matthews_corrcoef(y_test_re, y_pred_test_re.argmax(axis=1)))
        bal_arr.append(balanced_accuracy_score(y_test_re, y_pred_test_re.argmax(axis=1)))


    print("Accuracy: ", np.mean(acc_arr), np.std(acc_arr))
    print("F1-Score: ", np.mean(f1_arr), np.std(f1_arr))
    print("MCC: ", np.mean(mcc_arr), np.std(mcc_arr))
    print("Bal Acc: ", np.mean(bal_arr), np.std(bal_arr))



with tf.device('/gpu:0'):
    y_pred = model.predict(X_test)
    print("Classification Report Validation")
    cr = classification_report(y_test, y_pred.argmax(axis=1), output_dict = True)
    df = pd.DataFrame(cr).transpose()
    df.to_csv('results/CR_ANN_PB.csv')
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
    #print(matrix)
    print("F1 Score")
    print(f1_score(y_test, y_pred.argmax(axis=1), average = 'macro'))

'''

'''

