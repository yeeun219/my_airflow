import sys

import numpy as np
import pandas
from tensorflow.keras.layers import Dense
from tensorflow import keras

from keras import models
from keras import layers

class MarketingDataModeling:
    def __init__(self):
        pass

    def group_by_channel(self, data):
        grouped_by_channel= data.groupby('marketing_channel').mean('re')
        return grouped_by_channel

    def train_data_define(self,df):
        # train_data 생성
        y_train, y_test = df['marketing_channel'][:int(len(df) * 0.8)].to_numpy(), df['marketing_channel'][
                                                                                   int(len(df) * 0.8):].to_numpy()
        del (df['marketing_channel'])
        x_train, x_test = df[:int(len(df) * 0.8)].values, df[int(len(df) * 0.8):].values
        x_val = x_train[:int(len(x_train) * 0.9)]
        partial_x_train = x_train[int(len(x_train) * 0.9):]
        y_val = y_train[:int(len(y_train) * 0.9)]
        partial_y_train = y_train[int(len(y_train) * 0.9):]
        return y_train, y_test, x_train, x_test,x_val, partial_x_train,y_val,partial_y_train


    def assign_model1(self, df,y_train, y_test, x_train, x_test, x_val, partial_x_train, y_val, partial_y_train):

        model = keras.Sequential()
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        opt = keras.optimizers.Adam(learning_rate=0.005)
        model.compile(optimizer=opt,
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=20,
                            batch_size=512,
                            validation_data=(x_val, y_val))

        return history, model

    def assign_simpple_kerasModel(self, df, y_train, y_test, x_train, x_test, x_val, partial_x_train, y_val, partial_y_train):

        model2 = models.Sequential()
        model2.add(layers.Dense(16, activation='relu'))
        model2.add(layers.Dense(16, activation='relu'))
        model2.add(layers.Dense(1, activation='sigmoid'))
        model2.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        history = model2.fit(partial_x_train,
                            partial_y_train,
                            epochs=20,
                            batch_size=512,
                            validation_data=(x_val, y_val))
        return history,model2

    def perceptronModel(self, df, y_train, y_test, x_train, x_test, x_val, partial_x_train, y_val, partial_y_train):

        model3 = models.Sequential()
        model3.add(Dense(1, input_dim=3, activation='sigmoid'))
        # 3. 모델 학습과정 설정하기
        model3.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        # 4. 모델 학습시키기
        history = model3.fit(partial_x_train,
                            partial_y_train,
                            epochs=20,
                            batch_size=512,
                            validation_data=(x_val, y_val))

        return model3,history