# [실습]
#1. TRAIN 0.7 이상
#2. R2: 0.8 이상 / RMSE 사용

import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model        # model은 인풋 레이어를 명시해줘야함.
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import r2_score

#1. 데이터
# path = './_data/ddarung/'
# path = '../_data/ddarung/'
path = 'C:/study/_data/ddarung/'                            # 절대 경로
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    # 데이터에서 제외할 인덱스 칼럼이 0열에 있음. 열은 데이터가 아닌 인덱스라고 알려줘서 데이터로 들어가는 것을 방지.
                                                            # train_csv = pd.read_csv('./_data/ddarung/train.csv,' index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)  
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)                                      # (1459, 10) 카운트 분리하면 (1459, 9)

print(train_csv.columns)
                                                            # ['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                                                                #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                                                                #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],

print(train_csv.info())                                     # 결측치 데이터가 2개가 있음. 임의의 데이터를 넣으면 오차가 커지기 때문에 제거해야 한다. 단, 데이터가 적을시에는 삭제하는 것이 오히려 치명적이다.
print(test_csv.info())                                      # 카운트 없음.
print(train_csv.describe())

train_csv = train_csv.dropna()
print(train_csv.isnull().sum())                             # 결측값 확인
print(train_csv.shape)                                      # (1328, 10)

x = train_csv.drop(['count'], axis=1)                       # 작업할 때 axis가 1이면 행, 0이면 열
print(x)                                                    # [1459 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)                                              # (1459, )
                                                           
x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, test_size=0.2 
)                                                               # train은 대략 1167개 정도

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)                                             # x_train에 대한 범위의 가중치 생성
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
x_train = scaler.fit_transform(x_train)                         # 한 줄로 정리
x_test = scaler.transform(x_test)                               
test_csv = scaler.transform(test_csv)

# 2. 모델구성(순차형)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(9, )))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))                                 # activation 임의로 수정
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary()                                                         # 노드의 총 갯수는 drop을 해도 모두 동일하다.
                                                                        # drop을 해도 평가할 때는 전체 노드는 다 쓴다. / 훈련시에만 적용된다.
# Total params: 4,411         

# 2. 모델구성(함수형)
# input1 = Input(shape=(9, ))                                            
# Dense1 = Dense(50, activation='relu')(input1)
# drop1 = Dropout(0.5)(Dense1)
# Dense2 = Dense(40, activation='sigmoid')(drop1)
# drop2 = Dropout(0.3)(Dense2)
# Dense3 = Dense(30, activation='relu')(drop2)
# drop3 = Dropout(0.2)(Dense3)
# Dense4 = Dense(20, activation='linear')(Dense3)
# output1 = Dense(1, activation='linear')(Dense4)
# model = Model(inputs=input1, outputs=output1)                           # 시작하는 부분과 끝 부분이 어디인지 알려주는 부분
# model.summary()


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
                      filepath = filepath + 'k31_01' + date + '_' + filename
)

                                                                                       
model.fit(x_train, y_train, epochs=5000, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)                                                    # val_loss 즉, 검증할 때 손실값이 출력된다.
                                                                        # 기준을 잡을 때, val_loss로 기준을 잡는다.
                                                                        
# model.save(path + "keras31_ModelCheckPoint3_save_model.h5")
                                                                    
#4. 평가, 예측
print("=============== 1. 기본 출력 ========================")
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse)

y_predict = model.predict(x_test)
print("y_test(원래값):", y_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어:", r2)

# 제출할 놈
y_submit = model.predict(test_csv)
# print(y_submit)                                           # nan이 출력되는 걸로 봐선 test_csv에도 뭔가 결측치가 있었다는 뜻임.
# print(y_submit.shape)  

# .to_csv를 사용해서
# submission_0105.csv를 완성하시오!!!

# print(submission)
submission['count'] = y_submit                              # 대여수를 예측한 카운트에 submission에 넣어준다.
# print(submission)

submission.to_csv(path + 'submission_01122020.csv')         # 날짜.시간 01061152 = 1월 6일 11시 52분




'''
mse: 2350.1181640625
r2스코어: 0.6510410200214951


'''

'''
# 2. 모델구성(순차형)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(9, )))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))                                 # activation 임의로 수정
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='linear'))
model.add(Dense(1, activation='linear'))

mse: 2349.1318359375
r2스코어: 0.6511874466449943

'''

"""
CPU
# 2. 모델구성(순차형)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(9, )))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))                                 # activation 임의로 수정
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary()                        

mse: 2217.53857421875
r2스코어: 0.670727148244251

"""

"""
GPU

# 2. 모델구성(순차형)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(9, )))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))                                 # activation 임의로 수정
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary()                        

mse: 2108.220458984375
r2스코어: 0.678625272699176

"""