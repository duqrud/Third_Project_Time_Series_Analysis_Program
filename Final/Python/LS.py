# https://teddylee777.github.io/tensorflow/LSTM%EC%9C%BC%EB%A1%9C-%EC%98%88%EC%B8%A1%ED%95%B4%EB%B3%B4%EB%8A%94-%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90-%EC%A3%BC%EA%B0%80

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
import tensorflow as tf
from tensorflow import keras
import os
from math import sqrt
from sklearn.metrics import mean_squared_error
import tensorflow_addons as tfa
import tqdm
# window_size에 기반하여 20일 간의 데이터 셋을 묶어주는 역할
# 순차적으로 20일 동안의 데이터 셋을 묶고, 이에 맞는 label(예측 데이터)와 함게 예측

def make_dataset(data, label, window_size, dense):
    feature_list = []
    label_list = []
    rest_list = []
    for i in range(len(data) - window_size - dense):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size:i+window_size+dense]))
    for j in range(dense):
        rest_list.append(np.array(label.iloc[j+i+dense:j+i+window_size+dense]))
    return np.array(feature_list), np.array(label_list), np.array(rest_list)

def Train(stock_file, train_ratio, valid_ratio, window_size, dense, Model_size, patience, epochs, batch_size):
    # stock_file = pd.read_csv('01-삼성전자-주가.csv')

    df = stock_file
    
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    except:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    if int(df["Year"][0]) > df["Year"][len(df)-1]:
        df = df[::-1]

    # 정규화
    scaler = MinMaxScaler()
    scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_scaled = scaler.fit_transform(df[scale_cols])
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = scale_cols

    # window_size : 얼마동안(기간)의 주가 데이터에 기반하여 다음날 종가를 예측할 것인가
    test_part = int(len(df)*(1-train_ratio-valid_ratio))

    train = df_scaled[:-test_part]
    test = df_scaled[-test_part:]

    # feature와 label 정의
    feature_cols = ['Open', 'High', 'Low', 'Volume']
    label_cols = ['Close']

    train_feature = train[feature_cols]
    train_label = train[label_cols]

    # train dataset
    train_feature, train_label, train_rest_label = make_dataset(train_feature, train_label, window_size, dense)

    # train, validation set 생성
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_feature, train_label, test_size=valid_ratio)

    x_train.shape, x_valid.shape
    # ((6086, 20, 4), (1522, 20, 4)) 

    test_feature = test[feature_cols]
    test_label = test[label_cols]

    # test dataset (실제 예측 해볼 데이터)
    test_feature, test_label, test_rest_label = make_dataset(test_feature, test_label, window_size, dense)
    test_feature.shape, test_label.shape
    # ((180, 20, 4), (180, 1))

    # LSTML 모델 생성
    model = Sequential()
    model.add(LSTM(Model_size,
                input_shape=(train_feature.shape[1], train_feature.shape[2]),
                activation='relu',
                return_sequences=False
                )
            )
    model.add(Dense(dense))

    # Train 부분
    # 모델 학습
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=patience)

    tqdm_callback = tfa.callbacks.TQDMProgressBar()
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_valid, y_valid),
                        callbacks=[tqdm_callback, early_stop])

    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']

    x_len = np.arange(len(y_loss))
    plt.figure(4)
    plt.plot(y_vloss, marker='.', c='red', label="Validation-set Loss")
    plt.plot(y_loss, marker='.', c='blue', label="Train-set Loss")

    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # 예측
    pred = model.predict(test_feature)
    pred2 = [x[0] for x in pred]
    draw = [x[0] for x in test_label]

    rmse = sqrt(mean_squared_error(draw, pred2))
    rmse = round(rmse, 5)

    pred2.extend(pred[-1])

    del stock_file['Date']  # 위 줄과 같은 효과
    del stock_file['Year']  # 위 줄과 같은 효과
    del stock_file['Month']  # 위 줄과 같은 효과
    del stock_file['Day']  # 위 줄과 같은 효과

    stock_info = stock_file.values[1:].astype(np.float)  # 금액&거래량 문자열을 부동소수점형으로 변환한다
    price = stock_info[:, :-1]
    result = []
    
    for i in range(len(pred2)):
        pred2[i] = reverse_min_max_scaling(price, pred2[i])

    for i in range(len(draw)):
        draw[i] = reverse_min_max_scaling(price, draw[i])
    
    # # # 실제데이터와 예측한 데이터 시각화
    # 원래
    plt.figure(5)
    plt.plot(draw, label='actual')
    plt.plot(pred2, label='predict')
    plt.legend()
    plt.grid()
    plt.xlabel('시간')
    plt.ylabel('정규화 값')

    # 확대    
    plt.figure(6)
    plus_step = int(len(draw)*0.9)
    draw = draw[plus_step:]
    pred2 = pred2[plus_step:]
    plt.plot(draw, label='actual')
    plt.plot(pred2, label='predict')
    plt.legend()
    plt.grid()
    plt.xlabel('시간')
    plt.ylabel('정규화 값')
    
    '''
    학습양, RMSE
    '''
    return plt.figure(4), plt.figure(5), plt.figure(6), len(df) - test_part, rmse, model

def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

def Test(stock_file, expansion_ratio, window_size, dense, model):
    train_ratio = expansion_ratio*0.9
    valid_ratio = expansion_ratio*0.1
    df = stock_file
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    except:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    if int(df["Year"][0]) > df["Year"][len(df)-1]:
        df = df[::-1]

    # 정규화
    scaler = MinMaxScaler()
    scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_scaled = scaler.fit_transform(df[scale_cols])
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = scale_cols

    # window_size : 얼마동안(기간)의 주가 데이터에 기반하여 다음날 종가를 예측할 것인가
    test_part = int(len(df)*(1-train_ratio-valid_ratio))

    train = df_scaled[:-test_part]
    test = df_scaled[-test_part:]

    # feature와 label 정의
    feature_cols = ['Open', 'High', 'Low', 'Volume']
    label_cols = ['Close']

    train_feature = train[feature_cols]
    train_label = train[label_cols]

    # train dataset
    train_feature, train_label, train_rest_label = make_dataset(train_feature, train_label, window_size, dense)

    # train, validation set 생성
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_feature, train_label, test_size=valid_ratio)

    x_train.shape, x_valid.shape
    # ((6086, 20, 4), (1522, 20, 4))

    test_feature = test[feature_cols]
    test_label = test[label_cols]

    # test dataset (실제 예측 해볼 데이터)
    test_feature, test_label, test_rest_label = make_dataset(test_feature, test_label, window_size, dense)
    test_feature.shape, test_label.shape
    # ((180, 20, 4), (180, 1))

    # TEST 부분
    # model load
    
    
    
    # 예측
    pred = model.predict(test_feature)
    draw = [x[0] for x in test_label]
    pred2 = [x[0] for x in pred]
    pred2.extend(pred[-1][:dense])

    del stock_file['Date']  # 위 줄과 같은 효과
    del stock_file['Year']  # 위 줄과 같은 효과
    del stock_file['Month']  # 위 줄과 같은 효과
    del stock_file['Day']  # 위 줄과 같은 효과

    stock_info = stock_file.values[1:].astype(np.float)  # 금액&거래량 문자열을 부동소수점형으로 변환한다
    price = stock_info[:, :-1]
    result = []
    for i in range(len(pred[-1])):
        result.append(reverse_min_max_scaling(price, pred[-1][i]))
    
    for i in range(len(pred2)):
        pred2[i] = reverse_min_max_scaling(price, pred2[i])

    for i in range(len(draw)):
        draw[i] = reverse_min_max_scaling(price, draw[i])

    # # # 실제데이터와 예측한 데이터 시각화
    try:
        plt.clf()
    except:
        pass
    plt.figure(0)
    plt.plot(draw, label='actual')
    plt.plot(pred2, label='predict')
    plt.legend()
    # plt.show()
    # return plt.figure(0), pred[-1], result
    return plt.figure(0), result

# stock_file = pd.read_csv('../Data/AMZN.csv')
# model = tf.keras.models.load_model('../save/amaz1.h5')
# Test(stock_file, 0.95, 40, 25, model)