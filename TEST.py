import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, Dense, Dropout, Input, Softmax, MaxPooling2D, \
    AveragePooling2D, Conv2D, Conv2DTranspose, concatenate, BatchNormalization
from numpy import reshape
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
# from keras import initializers
from scipy.io import savemat, loadmat
import scipy.io as sio
import random
from collections.abc import Iterable
import matplotlib.pyplot as plt
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

matfn = 'F:/02POINT/02研究工作/博士/02导电机理（ERT）/last/train_dz.mat'
data = sio.loadmat(matfn)
train_X = data['train_dz']

matfn = 'F:/02POINT/02研究工作/博士/02导电机理（ERT）/last/train_lz.mat'
data = sio.loadmat(matfn)
train_Y = data['train_lz']

matfn = 'F:/02POINT/02研究工作/博士/02导电机理（ERT）/last/test_dz.mat'
data = sio.loadmat(matfn)
test_X = data['test_dz']

matfn = 'F:/02POINT/02研究工作/博士/02导电机理（ERT）/last/test_lz.mat'
data = sio.loadmat(matfn)
test_Y = data['test_lz']

# reshape
# train_X = train_X.reshape([37210, 208])
train_X = train_X.reshape([32602, 13, 16])
train_X = np.expand_dims(train_X, axis=3)

test_Y = test_Y.reshape([8150, 1600])
# test_Y = test_Y.reshape([9302, 40, 40])
# test_Y = np.expand_dims(test_Y, axis=3)

train_Y = train_Y.reshape([32602, 1600])
# train_Y = train_Y.reshape([37210, 40, 40])
# train_Y = np.expand_dims(train_Y, axis=3)

# test_X = test_X.reshape([9302, 208])
test_X = test_X.reshape([8150, 13, 16])
test_X = np.expand_dims(test_X, axis=3)
'''
index = [i for i in range(len(train_X))]
random.shuffle(index)
train_X = train_X[index]
train_Y = train_Y[index]
'''
X = train_X
Y = train_Y
X_SHAPE = X.shape
Y_SHAPE = Y.shape
print(X_SHAPE)
print(Y_SHAPE)

'''
#unet
def genModelSoft():

    inputs = Input((13, 16, 1))
    conv1 = Conv2D(4, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 1), strides=(2, 2), padding='same')(conv1)
    #Dropout(0.2),

    conv2 = Conv2D(8, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 1), strides=(2, 2), padding='same')(conv2)
    # Dropout(0.2),

    conv3 = Conv2D(16, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 1), strides=(2, 2), padding='same')(conv1)
    # Dropout(0.2),

    conv4 = Conv2D(24, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 1), strides=(2, 2), padding='same')(conv4)
    # Dropout(0.2),

    conv5 = Conv2D(32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(pool4)
    dConv4M = conv5
    #Dropout(0.2),

    dConv3 = Conv2DTranspose(24, kernel_size=(2, 2), strides=(2, 1), padding='same', activation='relu')(dConv4M)
    dConv3M = concatenate([dConv3, conv4], axis=3)
    # Dropout(0.2),

    dConv2 = Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 1), padding='same', activation='relu')(dConv3M)
    dConv2M = concatenate([dConv2, conv3], axis=3)
    # Dropout(0.2),

    dConv1 = Conv2DTranspose(8, kernel_size=(2, 2), strides=(2, 1), padding='same', activation='relu')(dConv2M)
    dConv1M = concatenate([dConv1, conv2], axis=3)
    # Dropout(0.2),

    dConv0 = Conv2DTranspose(4, kernel_size=(2, 2), strides=(2, 1), padding='same', activation='relu')(dConv1M)
    dConv0M = concatenate([dConv0, conv1], axis=3)
   # Dropout(0.2),

    dd1 = Conv2DTranspose(1, kernel_size=(2, 2), strides=(5, 5), padding='same', activation='relu')(dConv0M)
    dd2 = Conv2DTranspose(1, kernel_size=(2, 2), strides=(5, 5), padding='same', activation='relu')(dd1)
    dd3 = Conv2D(1, kernel_size=(2, 2), strides=(4, 4), padding='same', activation='relu')(dd2)
    #dd4 = Conv2D(1, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(dd3)
    model = Model(inputs, dd3)
    return model

#dnn
model = tf.keras.Sequential()
d1 = model.add(tf.keras.layers.Dense(units=400, input_dim=208, activation='relu')),
d2 = model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros')),
d3 = model.add(tf.keras.layers.Dense(units=500, activation='relu')),
d4 = model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros')),
d5 = model.add(tf.keras.layers.Dense(units=900, activation='relu')),
d6 = model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros')),
d7 = model.add(tf.keras.layers.Dense(units=400, activation='relu')),
d8 = model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros')),
d9 = model.add(tf.keras.layers.Dense(units=1000, activation='relu')),
d10 = model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros')),
d11 = model.add(tf.keras.layers.Dense(units=1600, activation='linear')),



#cnn
model = tf.keras.Sequential()
conv1 = model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu', input_shape=(13, 16, 1)))
pool1 = model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 2), padding='same'))
b1 = model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                                  moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
#model.add(tf.keras.layers.Dropout(0.1))

conv2 = model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
pool2 = model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=(2, 2), padding='same'))
b2 = model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                                  moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
#model.add(tf.keras.layers.Dropout(0.1))

conv3 = model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
pool3 = model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=(2, 2), padding='same'))
b3 = model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                                                  moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

conv4 = model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'))
pool4 = model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=(2, 2), padding='same'))

dConv3 = model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=(2, 2), strides=(5, 5), padding='same', activation='relu'))

dConv2 = model.add(tf.keras.layers.Conv2DTranspose(8, kernel_size=(2, 2), strides=(4, 4), padding='same', activation='relu'))

dConv1 = model.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='linear'))
#dConv2M = tf.keras.layers.concatenate([dConv2, conv3], axis=3)
#model.add(tf.keras.layers.Dropout(0.1))
'''

weight_decay = 0.0001

time_start = time.time()  # Start timing
def build_model():
    model = Sequential()

    model.add(Conv2D(4, (2, 2), padding='valid', activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=l2(weight_decay), input_shape=(13, 16, 1)))
    model.add(
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones',
                           moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None,
                           gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))
    model.add(Dropout(0.3))
    model.add(Conv2D(8, (2, 2), padding='valid', activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))
    model.add(Dropout(0.3))
    model.add(Conv2D(16, (2, 2), padding='valid', activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(960, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    model.add(Dropout(0.3))
    #model.add(Dense(1096, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    #model.add(Dropout(0.1))
    model.add(Dense(1280, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    model.add(Dropout(0.3))
    model.add(Dense(1600, activation='linear', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
    return model

   # global loss1


def lossFuncSoft(y_true, y_pred):
    k0 = 0.75
    loss1 = K.mean(K.square(y_pred - y_true)) * k0+K.mean(abs(y_pred - y_true)) * (1 - k0)
    total_loss = loss1
    return total_loss

'''
def build_model():
    model = Sequential()
    model.add(Conv2D(4, (2, 2), padding='valid', activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=l2(weight_decay), input_shape=(13, 16, 1)))
    model.add(MaxPooling2D((2, 1), strides=(2, 2)))
    model.add(Conv2D(8, (2, 2), padding='valid', activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D((2, 1), strides=(2, 2)))

    model.add(
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones',
                           moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None,
                           gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Conv2DTranspose(16, kernel_size=(2, 2), strides=(5, 5), padding='same', activation='relu'))
    model.add(Conv2DTranspose(1, kernel_size=(2, 2), strides=(4, 2), padding='same', activation='relu'))
    sgd = keras.optimizers.Adam(lr=0.00025, beta_1=0.888, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model
'''


model = build_model()
model.summary()
sgd = keras.optimizers.Adam(lr=0.0001, beta_1=0.888, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
model.compile(loss=lossFuncSoft, optimizer=sgd, metrics=['accuracy'])
history = model.fit(X, Y, batch_size=128, epochs=100)
time_end = time.time()  # The end of the timing
time_c = time_end - time_start  # Time taken to run

pd.DataFrame(history.history).to_csv('F:/02POINT/02研究工作/博士/02导电机理（ERT）/正演模拟数据/data/training_log_4.csv', index=False)
PP_result = model.predict(test_X, batch_size=32)
R_SHAPE = PP_result.shape
print(R_SHAPE)
PP_result = PP_result.reshape([326000, 40])
savemat('F:/02POINT/02研究工作/博士/02导电机理（ERT）/正演模拟数据/data/PP_4.mat', {'PP_result': PP_result})
# model.save('F:/02POINT/02研究工作/博士/02导电机理（ERT）/正演模拟数据/data/model0608.h5')

# loss curve
'''
df1 = pd.read_csv('training_log.csv')
epochs = range(len(df1['accuracy']))
fig = plt.figure(figsize=(10, 10))
ax1_1 = fig
y1_1 = df1['loss']
y1_2 = df1['accuracy']

plot1_1 = ax1_1.pl0t(epochs, y1_1, '-', marker='*', markersize=4, color='k', label='Loss')
ax1_2 = ax1_1.twinx()
plot1_2 = ax1_2.pl0t(epochs, y1_2, '-', marker='o', markersize=4, color='m', label='Accuracy')
ax1_1.set_xlabel('epoch')
ax1_1.set_ylabel('Loss')
ax1_2.set_ylabel('Accuracy')

lines = plot1_1 + plot1_2
# for tl in ax1_1.get_yticklabels():
#     tl.set_color('r')
for tl in ax1_1.get_xticklabels():
    tl.set_rotation(45)
    tl.set_fontsize(8)
# for tl in ax1_2.get_yticklabels():
#     tl.set_color([1, 0.647, 0])

plt.title('Model')
ax1_1.legend(lines, [l.get_label() for l in lines], loc=7)
plt.show()
print('accuracy')
'''

#  100 times output predictions

y_probas = np.stack([model(test_X, training=True)
                 for sample in range(100)])
y_samples_mean = y_probas.mean(axis=0)
y_samples_mean = y_samples_mean.reshape([326000, 40])
y_samples_std = y_probas.std(axis=0)
y_samples_std = y_samples_std.reshape([326000, 40])
#savemat('F:/02POINT/02研究工作/博士/02导电机理（ERT）/正演模拟数据/data/mc_d_10.mat', {'y_probas': y_probas})
savemat('F:/02POINT/02研究工作/博士/02导电机理（ERT）/正演模拟数据/data/mean_4.mat', {'y_samples_mean': y_samples_mean})
savemat('F:/02POINT/02研究工作/博士/02导电机理（ERT）/正演模拟数据/data/std_4.mat', {'y_samples_std': y_samples_std})
#model.save('F:/02POINT/02研究工作/博士/02导电机理（ERT）/正演模拟数据/data/model_10.h5')
#json_pre_0805 = model.to_json()
#model.save_weights('F:/02POINT/02研究工作/博士/02导电机理（ERT）/正演模拟数据/data/weights_10.h5')
print('end')
print('time cost', time_c, 's')



