import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar100

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)                                 # (50000, 32, 32, 3) (50000, 1)
                                                                    # 데이터 내용이나 순서의 영향을 받지 않는다. reshape
print(x_test.shape, y_test.shape)                                   # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))                       # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
                                                                    #   dtype=int64))
                                                                    
x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

x_train = x_train/255.
x_test = x_test/255.       
                                                                       
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D        # Maxpool이나 Pooing이나 똑같음
from tensorflow.keras.layers import Dropout

#2. 모델
model = Sequential()                                                            # 32*32*3
model.add(Dense(64, activation='relu', input_shape=(3072, )))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='linear'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))    
model.add(Dropout(0.3))                                                                            
model.add(Dense(100, activation='softmax'))                                      # arrayr가 0부터 9까지니 10개겠죠?
model.summary()

#3. 컴파일, 훈련    
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
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
                      filepath = filepath + 'k36_04' + date + '_' + filename
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

"""
기존 성능
loss: 0.3412693738937378
acc: 0.9643999934196472

padding 적용 후
loss: 0.39046329259872437
acc: 0.9668999910354614

Maxpool 적용 후
loss: 0.17730075120925903
acc: 0.9746999740600586

구글
loss: 0.1927846521139145
acc: 0.9732999801635742

dnn 적용
loss: 2.3026115894317627
acc: 0.1000000014901161

"""

