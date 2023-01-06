# [실습]
#1. TRAIN 0.7 이상
#2. R2: 0.8 이상 / RMSE 사용

import sklearn as sk
print(sk.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=1234
)

x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test,
    train_size=0.7, shuffle=True, random_state=1234
)

print(x)
print(x.shape)  # (506, 13)
print(y)
print(y.shape)  # (506, )

print(datasets.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']
print(datasets.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(40, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
import time     # 시간을 임포트
model.compile(loss='mse', optimizer='adam')     # 가급적 유사지표에서는 mse           
start = time.time()     # 시작 시간
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25)        # val_loss 즉, 검증할 때 손실값이 출력된다.
                                        # 기준을 잡을 때, val_loss로 기준을 잡는다.   
end = time.time()       # 종료 시간

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

print("============================")
print(y_test)
print(y_predict)
print("============================")

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)

print("걸린시간 :", end - start)

'''
최선의 판단

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(40))
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(16))
model.add(Dense(1))

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=1234
)

x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test,
    train_size=0.7, shuffle=True, random_state=1234
)

model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25) 

CPU 기준          
RMSE:  5.282456863022895
R2:  0.6897642169138176
val_loss: 44.2149

GPU 기준          
RMSE:  5.762504038570717
R2:  0.6308163530755024
val_loss: 50.2012

'''