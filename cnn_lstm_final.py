# -*- coding: utf-8 -*-
#!/usr/bin/env python

from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Dropout, Flatten, LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import input_data


# Convolution
kernel_size = [2, 3, 4]

# Training
time_steps = 4
batch_size = 16
epochs = 75

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true), axis=-1))
#载入模型
print('loading data...')

pems = input_data.read_data_sets()
speed_train = pems.train.speed
speed_train = speed_train.reshape((speed_train.shape[0], time_steps, 214, 1))
labels_train = pems.train.labels
speed_test = pems.test.speed
speed_test = speed_test.reshape((speed_test.shape[0], time_steps, 214, 1))
labels_test = pems.test.labels
speed_validation = pems.validation.speed
speed_validation = speed_validation.reshape((speed_validation.shape[0], time_steps, 214, 1))
labels_validation = pems.validation.labels

# 构建模型
print('build model...')
model = Sequential()

# 第一个卷积层
model.add(TimeDistributed(Conv1D(40,
                 kernel_size[1],
                 strides=1,
                 padding='valid'), input_shape=[time_steps, 214, 1]))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))

# 第二个卷积层
model.add(TimeDistributed(Conv1D(40, kernel_size[1], padding='valid')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))

# 第三个卷积层
model.add(TimeDistributed(Conv1D(40, kernel_size[0], padding='valid')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))

# 一维化
model.add(TimeDistributed(Flatten()))
model.add(Dense(214))
model.add(Dropout(0.5))

model.add(LSTM(64, return_sequences=True))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(LSTM(256))
model.add(Activation('tanh'))  
model.add(Dropout(0.5))
model.add(Dense(214))

model.summary()

model.load_weights("Model/cnn_lstm_finalForImputed.h5")

model.compile(loss='mean_squared_error', 
              optimizer='rmsprop',
              metrics=['mae', rmse,'cosine_similarity'])
"""
filepath = "Model/cnn_lstm_finalForImputed.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# 训练
print('Train...')
model.fit(speed_train, labels_train, 
#          validation_split=0.33,
          epochs=epochs, 
          batch_size=batch_size,
          callbacks=callbacks_list,
          validation_data=(speed_validation, labels_validation))
"""
score = model.evaluate(speed_test, labels_test, batch_size=batch_size)
print('Test score:', score)

"""
model_json = model.to_json()
with open("Model/cnn_lstm_finalForImputed.json", "w") as json_file:
    json_file.write(model_json)
print("Save model to disk")

json_file = open('Model/cnn_lstm_finalForImputed.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn_lstm_model = model_from_json(loaded_model_json)

cnn_lstm_model.load_weights("Model/cnn_lstm_finalForImputed.h5")
"""
