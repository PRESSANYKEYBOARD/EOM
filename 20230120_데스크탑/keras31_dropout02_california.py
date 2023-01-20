# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np                                                                  # 넘파이를 임포트시키고 간단하게 np로 명시해준다.
import tensorflow as tf                                                             # 텐서플로를 임포트시키고 tf라고 명시
from tensorflow.keras.models import Sequential, Model                               # Sequential은 레이어에 순차적으로 연산
from tensorflow.keras.layers import Dense, Input, Dropout                                    # Dense는 완전 연결층을 구현하는 레이어 모델
from sklearn.model_selection import train_test_split                                # 사이킷런(scikit-learn)의 model_selection 패키지 안에 train_test_split 모듈을 활용하여 손쉽게 train set(학습 데이터 셋)과 test set(테스트 셋)을 분리할 수 있다.
from sklearn.metrics import mean_squared_error, r2_score                            # RMSE 함수는 아직 없어서 직접 만들어 사용. - 회귀 분석 모델 / 사이킷런에서도 rmse는 제공하지 않음. / MSE 함수 불러옴.
                                                                                    # MSE보다 이상치에 덜 민감하다. 이상치에 대한 민감도가 MSE보단 적고 MAE보단 크기 때문에 이상치를 적절히 잘 다룬다고 간주되는 경향이 있다고 한다.
from sklearn.datasets import load_boston                                            # sklearn에서 제공하는 load_boston 가져오기
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = './_save/'
# path = '.._save/'
# path = 'c:/study/_save/'                                                          # 셋 다 모두 동일함.


from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model, load_model        # model은 인풋 레이어를 명시해줘야함.
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import r2_score

path = './_save/'
# path = '.._save/'
# path = 'c:/study/_save/'                                               # 셋 다 모두 동일함.

#1. 데이터
datasets = fetch_california_housing()
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

# 2. 모델구성(순차형)
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(8,)))
model.add(Dropout(0.5))
model.add(Dense(40, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary()                                                         # 노드의 총 갯수는 drop을 해도 모두 동일하다.
                                                                        # drop을 해도 평가할 때는 전체 노드는 다 쓴다. / 훈련시에만 적용된다.
# Total params: 4,361         



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
                                                                        
model.save(path+'keras31_dropout02_save_model.h5')                      # 가중치 및 모델 세이브
                                                                    
#4. 평가, 예측
print("=============== 1. 기본 출력 ========================")
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse)

y_predict = model.predict(x_test)
print("y_test(원래값):", y_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어:", r2)

"""
CPU
r2스코어: 0.6432356810322851

GPU
r2스코어: 0.6583654973086674

"""

