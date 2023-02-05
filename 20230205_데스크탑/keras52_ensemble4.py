# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np

#1. 데이터
x_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x_datasets.shape)                                                         # (100, 2) -> 삼전 시가+고가

y1 = np.array(range(2001, 2101))                                                 # (100,) -> 삼전 하루 뒤 '종가'
y2 = np.array(range(201, 301))                                                   # (100,) -> 아모레 하루 뒤 '종가'

# 실습!!! 만들어!!!

from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test, \
    y2_train, y2_test = train_test_split(
        x_datasets, y1, y2, train_size=0.7, random_state=1234
    )

print(x_train.shape, y1_train.shape, y2_train.shape)                            # (70, 2) (70,) (70,)
print(x_test.shape, y1_test.shape, y2_test.shape)                               # (30, 2) (30,) (30,)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1.
input1 = Input(shape=(2, ))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

#2-5. 모델5 분기1
dense5 = Dense(32, activation='relu', name='ds51')(output1)
dense5 = Dense(32, activation='relu', name='ds52')(dense5)
dense5 = Dense(32, activation='relu', name='ds53')(dense5)
output5 = Dense(33, activation='relu', name='ds54')(dense5)

#2.6 모델5 분기2
dense6 = Dense(32, activation='relu', name='ds61')(output1)
dense6 = Dense(32, activation='relu', name='ds62')(dense6)
dense6 = Dense(32, activation='relu', name='ds63')(dense6)
output6 = Dense(33, activation='relu', name='ds64')(dense6)

model = Model(inputs=[input1], outputs=[output5, output6])
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, [y1_train, y2_train], epochs=10, batch_size=8)

#4. 평가 예측
loss = model.evaluate([x_test], [y1_test, y2_test])
print("Loss: ", loss)


# loss가 왜 값이 3개가 나올까???

"""
Loss:  [2095525.375, 2065952.375, 29573.02734375, 1046.237548828125, 123.04573822021484]

"""