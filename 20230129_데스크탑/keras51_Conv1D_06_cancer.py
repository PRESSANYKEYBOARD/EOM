# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np                                                                  # 넘파이를 임포트시키고 간단하게 np로 명시해준다.
import tensorflow as tf                                                             # 텐서플로를 임포트시키고 tf라고 명시
from tensorflow.keras.models import Sequential, Model, load_model                               # Sequential은 레이어에 순차적으로 연산
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, Conv2D, Flatten, MaxPooling2D                                    # Dense는 완전 연결층을 구현하는 레이어 모델
from sklearn.model_selection import train_test_split                                # 사이킷런(scikit-learn)의 model_selection 패키지 안에 train_test_split 모듈을 활용하여 손쉽게 train set(학습 데이터 셋)과 test set(테스트 셋)을 분리할 수 있다.
from sklearn.metrics import mean_squared_error, r2_score                            # RMSE 함수는 아직 없어서 직접 만들어 사용. - 회귀 분석 모델 / 사이킷런에서도 rmse는 제공하지 않음. / MSE 함수 불러옴.
                                                                                    # MSE보다 이상치에 덜 민감하다. 이상치에 대한 민감도가 MSE보단 적고 MAE보단 크기 때문에 이상치를 적절히 잘 다룬다고 간주되는 경향이 있다고 한다.
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import Conv1D

path = './_save/'
# path = '.._save/'
# path = 'c:/study/_save/'                                                          # 셋 다 모두 동일함.

#1. 데이터
dataset= load_breast_cancer()
x = dataset.data                                                                    # 집에 대한 데이터
y = dataset.target                                                                  # 집 값

x_train, x_test, y_train, y_test = train_test_split(                                # 파라미터 (x와 y에 값을 대입) 
    x, y, train_size=0.7, random_state=123)                                         # 데이터셋에서 70%(x_train) # 데이터셋에서 30%(x_test) / 둘 중 하나만 명시해주면 된다. 
                                                                                    # test_size default=0.25
                                                                                    # shuffle=True 무작위 추출, False=순차적 추출 / default=True
                                                                                    # random_state로 잡아주면 그다음 데이터도 동일한 데이터로 들어감. 아무런 의미 없는 값을 넣어도 상관없다.

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
# (398, 30) (171, 30)

x_train = x_train.reshape(398, 30, 1)
x_test = x_test.reshape(171, 30, 1)

#2. 모델
model = Sequential()                               
# model.add(LSTM(units=64, input_shape=(4, 1), activation='relu'))
# model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(4, 1)))              # 81600
# model.add(GRU(64))                                                                          # LSTM이든 GRU이든 모두 가능하다.  
model.add(Conv1D(100, 2, input_shape=(30, 1)))   # 연산량 300
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
                                                              
'''
summray
node를 random하게 추출하여 훈련을 수행 -> 과적합 문제 해결
summary는 dropout된 node를 나누지 않음
predict 시에는 dropout 사용 X
Total params: 8,225
Trainable params: 8,225
Non-trainable params: 0

'''

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
                      filepath = filepath + 'k51_06' + date + '_' + filename
)
                                                                                       
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)                                                    # val_loss 즉, 검증할 때 손실값이 출력된다.
                                                                        # 기준을 잡을 때, val_loss로 기준을 잡는다.
                                                                        
model.save(path+'keras51_Conv1D06_save_model.h5')                      # 가중치 및 모델 세이브
                                                                    
#4. 평가, 예측
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
r2스코어: 0.806368539641875

lstm
r2스코어: 0.41539610465692267

Conv1D
r2스코어: 0.9027597455024796

"""

