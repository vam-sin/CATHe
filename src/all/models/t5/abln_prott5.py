#libraries
import pandas as pd 
from sklearn import preprocessing
import numpy as np 
import pickle
from sklearn.preprocessing import OneHotEncoder 
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
import math
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras import optimizers
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, Add, LSTM, Bidirectional, Reshape
from keras_self_attention import SeqSelfAttention
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, matthews_corrcoef, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import regularizers
from keras import backend as K
import keras
from sklearn.model_selection import KFold

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

filename = 'SF_Train_ProtT5.npz'
X_train = np.load(filename)['arr_0']
X_train = np.expand_dims(X_train, axis = 1)

# val
ds_val = pd.read_csv('Y_Val_SF.csv')

y_val = list(ds_val["SF"])

filename = 'SF_Val_ProtT5.npz'
X_val = np.load(filename)['arr_0']
X_val = np.expand_dims(X_val, axis = 1)

# test
ds_test = pd.read_csv('Y_Test_SF.csv')

y_test = list(ds_test["SF"])

filename = 'SF_Test_ProtT5.npz'
X_test = np.load(filename)['arr_0']
X_test = np.expand_dims(X_test, axis = 1)

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

# X, y = shuffle(X, y, random_state=42)
# print("Shuffled")

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("Conducted Train-Test Split")

# num_classes_train = len(np.unique(y_train))
# num_classes_test = len(np.unique(y_test))
# print(num_classes_train, num_classes_test)

# assert num_classes_test == num_classes_train, "Split not conducted correctly"

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

# test and train generators
# train_gen = bm_generator(X_train, y_train, bs)
# test_gen = bm_generator(X_test, y_test, bs)

# num_classes = 1707

# sensitivity metric
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Keras NN Model
def create_model():
    input_ = Input(shape = (1,1024,))
    x = Bidirectional(LSTM(256, activation = 'tanh', return_sequences = True))(input_)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256, activation = 'tanh', return_sequences = True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(256, activation = 'tanh', return_sequences = True))(x)
    x = Dropout(0.2)(x)
    x = SeqSelfAttention(attention_activation='sigmoid')(x)
    x = Flatten()(x)
    out = Dense(num_classes, activation = 'softmax')(x)
    classifier = Model(input_, out)

    return classifier

# training
num_epochs = 200

with tf.device('/gpu:0'):
    # model
    model = create_model()

    # adam optimizer
    opt = keras.optimizers.Adam(learning_rate = 1e-5)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy'])

    # callbacks
    mcp_save = keras.callbacks.ModelCheckpoint('saved_models/abln_prott5.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    callbacks_list = [reduce_lr, mcp_save]

    # test and train generators
    train_gen = bm_generator(X_train, y_train, bs)
    val_gen = bm_generator(X_val, y_val, bs)
    test_gen = bm_generator(X_test, y_test, bs)
    history = model.fit_generator(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(bs)), verbose=1, validation_data = val_gen, validation_steps = len(X_val)/bs, workers = 0, shuffle = True, callbacks = callbacks_list)
    model = load_model('saved_models/abln_prott5.h5', custom_objects=SeqSelfAttention.get_custom_objects(), compile=False)

    print("Validation")
    y_pred_val = model.predict(X_val)
    f1_score_val = f1_score(y_val, y_pred_val.argmax(axis=1), average = 'weighted')
    acc_score_val = accuracy_score(y_val, y_pred_val.argmax(axis=1))
    print("F1 Score: ", f1_score_val)
    print("Acc Score", acc_score_val)

    print("Testing")
    y_pred_test = model.predict(X_test)
    f1_score_test = f1_score(y_test, y_pred_test.argmax(axis=1), average = 'weighted')
    acc_score_test = accuracy_score(y_test, y_pred_test.argmax(axis=1))
    mcc_score = matthews_corrcoef(y_test, y_pred_test.argmax(axis=1))
    bal_acc = balanced_accuracy_score(y_test, y_pred_test.argmax(axis=1))
    print("F1 Score: ", f1_score_test)
    print("Acc Score: ", acc_score_test)
    print("MCC: ", mcc_score)
    print("Bal Acc: ", bal_acc)

with tf.device('/cpu:0'):
    y_pred = model.predict(X_test)
    print("Classification Report Validation")
    cr = classification_report(y_test, y_pred.argmax(axis=1), output_dict = True)
    df = pd.DataFrame(cr).transpose()
    df.to_csv('results/CR_ABLN_ProtT5.csv')
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
    print(matrix)
    print("F1 Score")
    print(f1_score(y_test, y_pred.argmax(axis=1), average = 'weighted'))

'''
/saved_models/abln_protbert.h5 (Beaker)
Validation
F1 Score:  0.9609832563891313
Acc Score 0.9609223929485433
Testing
F1 Score:  0.9608121715284709
Acc Score:  0.9607542448975473
MCC:  0.9590013496263621
Bal Acc:  0.9417163143702381
'''