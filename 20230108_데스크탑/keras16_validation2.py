# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np                                                                  # 넘파이를 임포트시키고 간단하게 np로 명시해준다.
from tensorflow.keras.models import Sequential                                      # Sequential은 레이어에 순차적으로 연산
from tensorflow.keras.layers import Dense                                           # Dense는 완전 연결층을 구현하는 레이어 모델

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# [실습] 잘라봐!!!         
x_train = x[0:10]                                                                   # x[:10]    # x[:-6]     동일한 값들이 나온다.
x_test = x[10:13]                                                                   # x[-6:-3] 
x_validation = x[13:16]                                                             # x[13:]    # x[-3:]
y_train = y[0:10]                                                                   # y[:10]    # y[:-6]
y_test = y[10:13]                                                                   # y[-6:-3] 
y_validation = y[13:16]                                                             # y[13:]    # y[-3:]

'''
모르면 이렇게 프린트 찍어서 나오는 거 보고 짜르면 된다.
print(x_train)
print(x_test)
print(x_validation)
print(y_train)
print(y_test)
print(y_validation)

'''

'''

#1. 데이터
x_train = np.array(range(1, 11))                                                    # 훈련 데이터
y_train = np.array(range(1, 11))        
x_test = np.array([11,12,13])                                                       # 평가 데이터
y_test = np.array([11,12,13])
x_validation = np.array([14,15,16])                                                 # 검증 데이터(문제지를 푼다.)
y_validation = np.array([14,15,16])

#2. 모델
model = Sequential()                                                                # 모델을 순차적으로 구성하겠다.
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
        validation_data=(x_validation, y_validation))                               # x에 대한 예상문제를 평가하는 과정을 추가 (validation_data)
                                                                                    # 데이터 검증 (훈련하고 검증하고)
                                                                                    # 훈련 + '검증(Validation)' + 평가 (fit + 'validation'+ evaluate)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

result = model.predict([17])
print("17의 예측값 :", result)

'''

'''
CPU: 
16.691515, loss 0.02499852515757084

GPU: 
16.99799, loss 1.1327598485877388e-06

'''