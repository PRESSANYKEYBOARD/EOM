# [실습]
#1. TRAIN 0.7 이상
#2. R2: 0.8 이상 / RMSE 사용

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model        # model은 인풋 레이어를 명시해줘야함.
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Conv1D

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)                                 # (60000, 28, 28) (60000,)
                                                                    # x에는 60000장의 흑백 데이터
                                                                    # 데이터 내용이나 순서의 영향을 받지 않는다. reshape
print(x_test.shape, y_test.shape)                                   # (10000, 28, 28) (10000,)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train/255.
x_test = x_test/255.

#2. 모델구성
model = Sequential()                               
# model.add(LSTM(units=64, input_shape=(4, 1), activation='relu'))
# model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(4, 1)))              # 81600
# model.add(GRU(64))                                                                          # LSTM이든 GRU이든 모두 가능하다.  
model.add(Conv1D(100, 2, input_shape=(28, 28)))                                                 # 연산량 300
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='softmax'))

#3. 컴파일, 훈련    
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', 
                            #   restore_best_weights=False,
                              verbose=1)

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
                      filepath = filepath + 'k51_11' + date + '_' + filename
)
                                                                                       
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)                                                    # val_loss 즉, 검증할 때 손실값이 출력된다.
                                                                        # 기준을 잡을 때, val_loss로 기준을 잡는다.
                                                                        
# model.save(path + "keras31_ModelCheckPoint3_save_model.h5")

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss:', results[0])                                                     # loss 값 반환
print('acc:', results[1])                                                      # acc 값 반환

# es, mcp 적용 / val 적용
