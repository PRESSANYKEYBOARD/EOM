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
    train_size=0.7, shuffle=True, random_state=64
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
model.add(Dense(40))
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(16))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
model.fit(x_train, y_train, epochs=640, batch_size=64)

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


'''
최선의 판단

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(110))
model.add(Dense(180))
model.add(Dense(275))
model.add(Dense(320))
model.add(Dense(160))
model.add(Dense(130))
model.add(Dense(80))
model.add(Dense(65))
model.add(Dense(30))
model.add(Dense(1))

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=64
)
 
model.fit(x_train, y_train, epochs=3200, batch_size=100)
RMSE:  4.390121377597739
R2:  0.7967169945329375

# 아 몰랑 R2 0.8 어케했누!!! 자정 5분전까지 했는데도 안 나온다...

'''