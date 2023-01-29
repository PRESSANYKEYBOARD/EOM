# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model        # model은 인풋 레이어를 명시해줘야함.
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, Conv2D, Flatten, MaxPooling2D   
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Conv1D

#1. 데이터
# path = './_data/bike/'
# path = '../_data/bike/'
path = 'C:/study/_data/bike/'                                   # 절대 경로
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # 데이터에서 제외할 인덱스 칼럼이 0열에 있음. 열은 데이터가 아닌 인덱스라고 알려줘서 데이터로 들어가는 것을 방지.
                                                                # train_csv = pd.read_csv('./_data/ddarung/train.csv,' index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)           # index_col 지정하지 않으면, 인덱스가 자동으로 생성된다.
submission = pd.read_csv(path + 'samplesubmission.csv', index_col=0)

print(train_csv)                                                # [10886 rows x 11 columns]
print(train_csv.shape)                                          # (10886, 11)
print(submission.shape)                                         # (6493, 1)

print(train_csv.columns)
                                                                # (['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                                                                    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
                                                                    #   dtype='object')
print(train_csv.info()) 
print(test_csv.info()) 
print(train_csv.describe())                                     # [8 rows x 11 columns]

#### 결측치 처리 1. 제거 ####
print(train_csv.isnull().sum())                                 # 중단점 클릭하고 F5 하면 중단점 직전까지 실행이 된다.

'''
결측치 데이터
season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
casual        0
registered    0
count         0
'''

train_csv = train_csv.dropna()
print(train_csv.isnull().sum())                                     # 결측값 확인
print(train_csv.shape)  # (10886, 11)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)       # 작업할 때 axis가 1이면 행, 0이면 열
                                                                    # train.csv와 test.csv를 비교했을 때 casual, registered, count가 있고 없고의 차이임. 이걸 제거해주는 작업임.

print(x)                                                            # [10886 rows x 10 columns]
y = train_csv['count']
print(y)                                                            # (10886, 11)
print(y.shape)                                                      # (10886,)
                                                           
x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, test_size=0.2 
)                                                               # train은 대략 8,708개 정도

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)                                             # x_train에 대한 범위의 가중치 생성
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
x_train = scaler.fit_transform(x_train)                         # 한 줄로 정리
x_test = scaler.transform(x_test)                               
test_csv = scaler.transform(test_csv)

print(x_train.shape, x_test.shape, test_csv.shape)            # (8708, 8) (2178, 8) (6493, 8)

x_train = x_train.reshape(8708, 8, 1)
x_test = x_test.reshape(2178, 8, 1)

#2. 모델
model = Sequential()                               
# model.add(LSTM(units=64, input_shape=(4, 1), activation='relu'))
# model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(4, 1)))              # 81600
# model.add(GRU(64))                                                                          # LSTM이든 GRU이든 모두 가능하다.  
model.add(Conv1D(100, 2, input_shape=(8, 1)))   # 연산량 300
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

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
                      filepath = filepath + 'k51_05' + date + '_' + filename
)
                                                                                       
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)                                                    # val_loss 즉, 검증할 때 손실값이 출력된다.
                                                                        # 기준을 잡을 때, val_loss로 기준을 잡는다.
                                                                        
model.save(path+'keras51_Conv1D05_save_model.h5')                      # 가중치 및 모델 세이브
                                                                    
#4. 평가, 예측
print("=============== 1. 기본 출력 ========================")
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse)

y_predict = model.predict(x_test)
print("y_test(원래값):", y_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어:", r2)

# training이 진행된 모델이 conv2D로 시작되어 input_shape이 4차원으로 이뤄져야 함
# predict에 대입될 데이터도 만들어놓은 모델을 이용하므로 4차원으로 reshape 해줘야 함

# 제출할 놈
y_submit = model.predict(test_csv.reshape(6493, 8, 1))       # training이 진행된 모델이 conv2D로 시작되어 input_shape이 4차원으로 이뤄져야 함
                                                            # predict에 대입될 데이터도 만들어놓은 모델을 이용하므로 4차원으로 reshape 해줘야 함
# print(y_submit)                                           # nan이 출력되는 걸로 봐선 test_csv에도 뭔가 결측치가 있었다는 뜻임.
# print(y_submit.shape)  

# .to_csv를 사용해서
# submission_0105.csv를 완성하시오!!!

# print(submission)
submission['count'] = y_submit                              # 대여수를 예측한 카운트에 submission에 넣어준다.
# print(submission)

submission.to_csv(path + 'submission_01292052.csv')         # 날짜.시간 01061152 = 1월 6일 11시 52분

"""
CPU
r2스코어: 0.5981462497416854

GPU
r2스코어: 0.602071834685262

dnn
r2스코어: 0.6476696393639259

cnn
r2스코어: 0.46328707894465904

Conv1D
r2스코어: 0.32905958424340165

"""