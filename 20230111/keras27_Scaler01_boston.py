# [실습]
#1. TRAIN 0.7 이상
#2. R2: 0.8 이상 / RMSE 사용

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

# print("최소값:", np.min(x))                             # 0
# print("최대값:", np.max(x))                             # 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    random_state=333,
    test_size=0.2
    # stratify=y
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)                                           # x_train에 대한 범위의 가중치 생성
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x_train = scaler.fit_transform(x_train)                     # 31.32번 라인의 내용을 한 줄로 정리

#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(13,)))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(120, activation='relu'))
model.add(Dense(30, activation='linear'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
import time                                                                             # 시간을 임포트
model.compile(loss='mse', optimizer='adam',     
             metrics=['mae'])
                         
start = time.time()                                                                     # 시작 시간
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          verbose=1)                                                  # val_loss 즉, 검증할 때 손실값이 출력된다.
                                                                      # 기준을 잡을 때, val_loss로 기준을 잡는다.   
end = time.time()                                                     # 종료 시간

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse)
print('mae:', mae)

y_predict = model.predict(x_test)

print("y_test(원래값):", y_test)
r2 = r2_score(y_test, y_predict)
print(r2)

print("걸린시간 :", end - start)


'''
MinMaXScaler

CPU

MSE: 18.52433967590332
MAE: 3.0683512687683105
걸린시간: 3.8335773944854736

GPU

MSE: 18.474079132080078
MAE: 3.1125190258026123
걸린시간: 3.828256845474243

...
...

StandardScaler

CPU

MSE: 17.66674995422363
MAE: 2.6835546493530273
걸린시간: 3.8174784183502197

GPU

MSE: 21.17184066772461
MAE: 2.9273364543914795
걸린시간: 3.7656490802764893

'''