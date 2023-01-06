# [괒[, 실습]
# R2 0.62 이상

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=1234
)

x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test,
    train_size=0.7, shuffle=True, random_state=1234
)

print(x)
print(x.shape)  # (442, 10)
print(y)
print(y.shape)  # (442, )

print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=10))
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
값 조정

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=1234
)

x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test,
    train_size=0.7, shuffle=True, random_state=1234
)

model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25)
          
CPU 기준          
RMSE:  62.24204605634982
R2:  0.3114132279369527
val_loss: 3367.6467
걸린시간: 27.97167468070984

GPU 기준          
RMSE:  65.27239469404677
R2:  0.24273122982537165
val_loss: 50.2012
걸린시간: 26.33858299255371

'''