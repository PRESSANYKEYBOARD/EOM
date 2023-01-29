# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()                                               #교육용 자료, 이미 train/test 분류

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/255
x_test = x_test/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D                                # Maxpool이나 Pooing이나 똑같음

# 2. 모델 구성 
model = Sequential()
model.add(Conv2D(filters=256, 
                 kernel_size=(3, 3), 
                 padding='same',
                 input_shape=(32, 32, 3), 
                 activation='relu'))                                                                    # (27, 27, 128)
model.add(MaxPooling2D())
model.add(Conv2D(filters=128, kernel_size=(2, 2)))                                                      # (26, 26, 64)
model.add(Conv2D(filters=64, kernel_size=(2, 2)))                                                       # (25, 25, 64)
model.add(Flatten())
model.add(Dense(32, activation='relu'))                                                                 # input_shape=(60000, 40000) 6만 4만 인풋이야
model.add(Dense(24, activation='relu')) 
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련    
model.compile(loss='mse', optimizer='adam',
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
                      filepath = filepath + 'k35_03' + date + '_' + filename
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
GPU 기준
loss: 0.062199853360652924
acc: 0.6172999739646912

"""