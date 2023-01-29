# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

path = './_save/'
# path = '.._save/'
# path = 'c:/study/_save/'                                               # 셋 다 모두 동일함.

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']

# print("최소값:", np.min(x))                                           # 0
# print("최대값:", np.max(x))                                           # 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    random_state=333,
    test_size=0.2
    # stratify=y
)

scaler = MinMaxScaler()                                                 # x_train에 대한 범위의 가중치 생성
x_train = scaler.fit_transform(x_train)                     
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)                                      # (1437, 64) (360, 64)

x_train = x_train.reshape(1437, 8, 8, 1)
x_test = x_test.reshape(360, 8, 8, 1)
print(x_train.shape, x_test.shape)                                      # (1437, 8, 8, 1) (360, 8, 8, 1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(8, 8, 1)))
model.add(Flatten()) #dropout 훈련 시에만 적용된다.
model.add(Dense(24, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'linear'))
model.add(Dense(4, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))

# #2. 모델구성
# model = Sequential()
# model.add(Dense(50, activation='linear', input_shape=(30,)))            # input_dim=30
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))                               # 0과 1 사이의 값을 찾아야 되기 때문에 sigmoid로 맞춰준다.
#                                                                         # 2진 분류네???                                                                   

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',     
             metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                            #   restore_best_weights=False,
                              verbose=1, )

import datetime                                                                 # 데이터 타임 임포트해서 명시해준다.
date = datetime.datetime.now()                                                  # 현재 날짜와 시간이 date로 반환된다.

print(date)                                                                     # 2023-01-12 14:57:54.345489
print(type(date))                                                               # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")                                               # date를 str(문자형)으로 바꾼다.
                                                                                # 0112_1457
print(date)
print(type(date))                                                               # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'                                    # -는 연산 -가 아니라 글자이다. str형이기 때문에...
                                                                                # 0037-0.0048.hdf5

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                    #   filepath=path + 'MCP/keras31_ModelCheckPoint3.hdf5')
                      filepath = filepath + 'k39_09' + date + '_' + filename
)
                                                                                       
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)                                                    # val_loss 즉, 검증할 때 손실값이 출력된다.
                                                                        # 기준을 잡을 때, val_loss로 기준을 잡는다.
                                                                        
model.save(path+'keras39_dropout09_save_model.h5')                      # 가중치 및 모델 세이브

#4. 평가 ,예측
print("=============== 1. 기본 출력 ========================")
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse)

y_predict = model.predict(x_test)
print("y_test(원래값):", y_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어:", r2)

"""
CPU(minmax) 
0.8539258721528856
걸린시간 : 3.7504191398620605

GPU(minmax) 
0.8290388810289951
걸린시간 : 9.980056285858154

CPU(standard) 
0.7465749115700764
걸린시간 : 3.8035295009613037

GPU(standard) 
0.7794653278151802
걸린시간 : 10.22622299194336

cnn
r2스코어: 0.9343823601480101

"""
