# [실습]
# R2 0.55~0.6 이상

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=1234
)

x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test,
    train_size=0.7, shuffle=True, random_state=1234
)

print(x)
print(x.shape)  # (20640, 8)
print(y)
print(y.shape)  # (20640, )

print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
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

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=1234
)

x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test,
    train_size=0.7, shuffle=True, random_state=1234
)

하드웨어 사양 기준(노트북)

CPU: Intel Core i7-8750H  2.2Ghz (6C12T)
GPU: Nvidia Geforce GTX1060 6GB(notebook)
RAM: Samsung DDR4-3200 16GB x2 = 32GB

테스트 기준

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

model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25) 

CPU 기준          
RMSE:  0.6811830529600216
R2:  0.6897642169138176
val_loss: 44.2149
걸린시간: 69.82509565353394

GPU 기준          
RMSE: 0.6666081678883696
R2:  0.6697907216352472
val_loss: 0.4881
걸린시간: 64.57769513130188

'''