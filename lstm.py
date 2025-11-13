# -*- coding: utf-8 -*-
#!/usr/bin/env python
from __future__ import print_function

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import input_data

# ===================== 超参数设置 =====================
BATCH_SIZE = 64
TIME_STEPS = 4
INPUT_SIZE = 214
OUTPUT_SIZE = 214

# ===================== 自定义评估函数 =====================
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# ===================== 构建模型函数 =====================
def build_model():
    print('Building LSTM model...')
    model = Sequential()
    model.add(LSTM(input_shape=(TIME_STEPS, INPUT_SIZE),
                   units=64,
                   return_sequences=True))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(LSTM(units=256))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(OUTPUT_SIZE))

    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['mae', rmse, 'cosine_similarity'])
    return model

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    print('Loading data...')
    train, validation, test = input_data.read_data_sets()

    speed_train = train.speed
    labels_train = train.labels
    speed_validation = validation.speed
    labels_validation = validation.labels
    speed_test = test.speed
    labels_test = test.labels

    # 构建模型
    model = build_model()

    # 尝试加载已有权重（可选）
    try:
        model.load_weights("Model/lstmForImputed.h5")
        print("Loaded weights from Model/lstm.h5")
    except Exception as e:
        print("No existing weights found, training from scratch.")

    # 设置回调
    filepath = "Model/lstmForImputed.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    print('Training...')
    model.fit(speed_train, labels_train,
              epochs=300,
              batch_size=BATCH_SIZE,
              validation_data=(speed_validation, labels_validation),
              callbacks=callbacks_list)

    # 保存模型结构
    model_json = model.to_json()
    with open("Model/lstmForImputed.json", "w") as json_file:
        json_file.write(model_json)
    print("Model saved to disk.")

# ===================== 导入时暴露的变量 =====================
# 注意：下面这部分在 import 时会执行，但不会触发训练
try:
    json_file = open('Model/lstmForImputed.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("Model/lstmForImputed.h5")
    print("Pre-trained LSTM model loaded successfully.")
except Exception as e:
    print("Warning: cannot load pretrained model:", e)
    model = build_model()
