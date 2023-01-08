# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np                                                                  # 넘파이를 임포트시키고 간단하게 np로 명시해준다.
from tensorflow.keras.models import Sequential                                      # Sequential은 레이어에 순차적으로 연산
from tensorflow.keras.layers import Dense                                           # Dense는 완전 연결층을 구현하는 레이어 모델
from sklearn.model_selection import train_test_split                                # 사이킷런(scikit-learn)의 model_selection 패키지 안에 train_test_split 모듈을 활용하여 손쉽게 train set(학습 데이터 셋)과 test set(테스트 셋)을 분리할 수 있다.

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x, y,                           # 먼저, x와 y의 데이터를 받아서 0.2로 자른다.
    test_size=0.2, shuffle=False                                                    # 값이 순차적으로 나와야 되기 때문에 shuffle을 False로 설정.
)

'''
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test,       # 그 다음, x_test와 y_test의 데이터를 받아서 0.5의 비율로 동일하게 자른다.
    train_size=0.5, shuffle=False                                                   # 값이 순차적으로 나와야 되기 때문에 shuffle을 False로 설정.
)
                                                                                    # 이렇게 train에서 val을 한번 더 분리해주기보다 아래의 model.fit에서 validation_split 사용하기.
'''



'''
# [실습] 잘라봐!!!         
x_train = x[0:10]                                                                   # x[:10]    # x[:-6]     동일한 값들이 나온다.
x_test = x[10:13]                                                                   # x[-6:-3] 
x_validation = x[13:16]                                                             # x[13:]    # x[-3:]
y_train = y[0:10]                                                                   # y[:10]    # y[:-6]
y_test = y[10:13]                                                                   # y[-6:-3] 
y_validation = y[13:16]                                                             # y[13:]    # y[-3:]

'''

'''
#1. 데이터
x_train = np.array(range(1, 11))                                                    # 훈련 데이터
y_train = np.array(range(1, 11))        
x_test = np.array([11,12,13])                                                       # 평가 데이터
y_test = np.array([11,12,13])
x_validation = np.array([14,15,16])                                                 # 검증 데이터(문제지를 푼다.)
y_validation = np.array([14,15,16])

'''

#2. 모델
model = Sequential()                                                                # 모델을 순차적으로 구성하겠다.
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25)                                                    # x에 대한 예상문제를 평가하는 과정을 추가 (validation_data)
                                                                                    # 데이터 검증 (훈련하고 검증하고)
                                                                                    # 훈련 + '검증(Validation)' + 평가 (fit + 'validation'+ evaluate)
                                                                                    # validation_data를 통해서 val_loss 추가 / val_loss 즉, 검증할 때 손실값이 출력된다. / 기준을 잡을 때, val_loss로 기준을 잡는다.
                                                                                    # 훈련(train)보다 검증(validation)결과를 기준으로 테스트 결과를 판단해야 함.
                                                                                    # validation_split을 통해서 x_train과 y_train 중 0.25의 validation 값 지정.
                                                                                    

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

result = model.predict([17])
print("17의 예측값 :", result)



'''
CPU: 
16.169634, loss 0.3516864776611328

GPU: 
16.070143, loss 0.5226377248764038

'''