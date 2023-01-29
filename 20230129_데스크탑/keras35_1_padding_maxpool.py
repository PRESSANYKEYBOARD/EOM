# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)                                 # (60000, 28, 28) (60000,)
                                                                    # x에는 60000장의 흑백 데이터
                                                                    # 데이터 내용이나 순서의 영향을 받지 않는다. reshape
print(x_test.shape, y_test.shape)                                   # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape)                                 # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)                                   # (10000, 28, 28, 1) (10000,)

print(np.unique(y_train, return_counts=True))                       # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
                                                                    #   dtype=int64))
                                                                    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D        # Maxpool이나 Pooing이나 똑같음

#2. 모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28, 28, 1), 
                 padding='same',    # vaild
                 strides=3,
                 activation='relu' ))                                           # (28, 28, 128)
model.add(MaxPooling2D())                                                       # (14, 14, 128)
                                                                                # 28에서 14로 반을 잘라버림.
                                                                                # 연산량 없이 압축이 가능하다.
                                                                                # maxpool : 커널 사이즈로 나눴을 때, 가장 값이 큰 부분만 빼서 계산
model.add(Conv2D(filters=64, kernel_size=(2,2), 
                 padding='same'))                                               # (26, 26, 64)
model.add(Conv2D(filters=64, kernel_size=(2,2)))                                # (25, 25, 64) → flatten 40000
model.add(Flatten())                                                            # 40000
model.add(Dense(32, activation='relu'))                                         # input_shape = (40000, )
                                                                                # (6만, 4만)이 인풋이야.
                                                                                # 6만 = batch_size, 4만 = input_dim
                                                                                
model.add(Dense(10, activation='softmax'))                                      # arrayr가 0부터 9까지니 10개겠죠?

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
                      filepath = filepath + 'k35_01' + date + '_' + filename
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
결과:
loss: 2.3009276390075684
acc: 0.11349999904632568

"""