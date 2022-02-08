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

X_train_df = list(pd.read_csv('../../data/final/CSV/Train.csv')["Sequence"])
X_train_other_df = list(pd.read_csv('../../data/other/protbert_data/Other_Train_SF_US_New.csv')["Sequence"])
# print(X_train_df)

X_train = []

for i in X_train_df:
    X_train.append(len(i))

for i in X_train_other_df:
    X_train.append(len(i))

X_test_df = list(pd.read_csv('../../data/final/CSV/Test.csv')["Sequence"])
X_test_other_df = list(pd.read_csv('../../data/other/protbert_data/Other_Test_SF_US_New.csv')["Sequence"])
# print(X_train_df)

X_test = []

for i in X_test_df:
    X_test.append(len(i))

for i in X_test_other_df:
    X_test.append(len(i))

X_val_df = list(pd.read_csv('../../data/final/CSV/Val.csv')["Sequence"])
X_val_other_df = list(pd.read_csv('../../data/other/protbert_data/Other_Val_SF_US_New.csv')["Sequence"])
# print(X_train_df)

X_val = []

for i in X_val_df:
    X_val.append(len(i))

for i in X_val_other_df:
    X_val.append(len(i))

ds_train = pd.read_csv('../../data/final/CSV/Train.csv')

y_train = list(ds_train["SF"])

for i in range(len(X_train_other_df)):
    y_train.append('other')

ds_val = pd.read_csv('../../data/final/CSV/Val.csv')

y_val = list(ds_val["SF"])

for i in range(len(X_val_other_df)):
    y_val.append('other')

ds_test = pd.read_csv('../../data/final/CSV/Test.csv')

y_test = list(ds_test["SF"])

for i in range(len(X_test_other_df)):
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

df = pd.DataFrame(list(zip(X_train, y_train)),
               columns =['Length', 'SF'])

print(df)

y_train_un = list(set(y_train))

print(len(y_train_un))
y_train_un = np.sort(y_train_un)
for i in y_train_un:
    df1 = df[df['SF'] == i]
    le = list(df1["Length"])
    print(i, np.mean(le), np.std(le))

X_train, y_train = shuffle(X_train, y_train, random_state=42)
print("Shuffled")

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
    input_ = Input(shape = (1,))
    x = Dense(1024, activation = "relu")(input_)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation = 'softmax')(x)
    classifier = Model(input_, out)

    return classifier

# training
num_epochs = 10

with tf.device('/gpu:0'):
    # model
    model = create_model()

    # adam optimizer
    opt = keras.optimizers.Adam(learning_rate = 1e-5)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy', sensitivity])

    # callbacks
    mcp_save = keras.callbacks.ModelCheckpoint('ann_length.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)
    callbacks_list = [reduce_lr, mcp_save, early_stop]

    # test and train generators
    train_gen = bm_generator(X_train, y_train, bs)
    val_gen = bm_generator(X_val, y_val, bs)
    test_gen = bm_generator(X_test, y_test, bs)
    #history = model.fit_generator(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(bs)), verbose=1, validation_data = val_gen, validation_steps = len(X_val)/bs, workers = 0, shuffle = True, callbacks = callbacks_list)
    model = load_model('ann_length.h5', custom_objects={'sensitivity':sensitivity})

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

    print("Bootstrapping Results")
    num_iter = 1000
    f1_arr = []
    acc_arr = []
    mcc_arr = []
    bal_arr = []
    for it in range(num_iter):
        print("Iteration: ", it)
        X_test_re, y_test_re = resample(X_test, y_test, n_samples = len(y_test), random_state=it)
        y_pred_test_re = model.predict(X_test_re)
        # print(y_test_re)
        f1_arr.append(f1_score(y_test_re, y_pred_test_re.argmax(axis=1), average = 'macro'))
        acc_arr.append(accuracy_score(y_test_re, y_pred_test_re.argmax(axis=1)))
        mcc_arr.append(matthews_corrcoef(y_test_re, y_pred_test_re.argmax(axis=1)))
        bal_arr.append(balanced_accuracy_score(y_test_re, y_pred_test_re.argmax(axis=1)))


    print("Accuracy: ", np.mean(acc_arr), np.std(acc_arr))
    print("F1-Score: ", np.mean(f1_arr), np.std(f1_arr))
    print("MCC: ", np.mean(mcc_arr), np.std(mcc_arr))
    print("Bal Acc: ", np.mean(bal_arr), np.std(bal_arr))

with tf.device('/cpu:0'):
    y_pred = model.predict(X_test)
    print("Classification Report Validation")
    cr = classification_report(y_test, y_pred.argmax(axis=1), output_dict = True)
    df = pd.DataFrame(cr).transpose()
    df.to_csv('CR_ANN_Length.csv')
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
    print(matrix)
    print("F1 Score")
    print(f1_score(y_test, y_pred.argmax(axis=1), average = 'weighted'))

'''
Sequence Length Classifier
Accuracy:  0.03338195861264937 0.0021798833044140726
F1-Score:  0.0004445062387185485 4.514739270775787e-05
MCC:  0.02735765816420239 0.0023156411155433976
Bal Acc:  0.002332759793150367 0.00010639671446025988
'''