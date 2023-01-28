# 49_2 복붙
# boston부터 cifar100까지

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model, load_model        # model은 인풋 레이어를 명시해줘야함.
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Conv1D

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

print(x_train.shape, x_test.shape)                                      # (16512, 8) (4128, 8)

x_train = x_train.reshape(16512, 8, 1, 1)
x_test = x_test.reshape(4128, 8, 1, 1)
print(x_train.shape, x_test.shape)                                      # (16512, 2, 2, 2) (4128, 2, 2, 2)

#2. 모델구성
model = Sequential()                               
# model.add(LSTM(units=64, input_shape=(4, 1), activation='relu'))
# model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(4, 1)))              # 81600
# model.add(GRU(64))                                                                          # LSTM이든 GRU이든 모두 가능하다.  
model.add(Conv1D(100, 2, input_shape=(8, 1)))                                                 # 연산량 300
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

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
                      filepath = filepath + 'k51_02' + date + '_' + filename
)
                                                                                       
model.fit(x_train, y_train, epochs=100, batch_size=32,
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

"""
결과는???


"""