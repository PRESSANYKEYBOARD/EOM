import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score


#1. 데이터
datasets = load_diabetes()                                      # 소문자는 함수, 대문자는 클래스
x = datasets.data
y = datasets.target
print(x.shape, y.shape)                                         # (442, 10) (442, )
                                                                # 행 무시, 열 우선 / input_dim = 10
                                                           
x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, test_size=0.2 
)                                                               # train은 대략 353개 정도

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)                                           # x_train에 대한 범위의 가중치 생성
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x_train = scaler.fit_transform(x_train)                     # 한 줄로 정리

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_shape=(10, )))                         # 스칼라가 10개로 볼 수 있기 때문에, (10, ) 라고 해석할 수 있다.
                                                                # (100, 10, 5) 라는 데이터가 들어가면, 행 무시 열 우선이기 때문에 (10, 5)로 들어감.
                                                                # 앞으로는 이렇게 쓰자!!!
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(120, activation='relu'))
model.add(Dense(30, activation='linear'))
model.add(Dense(1))


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

MSE: 2727.667724609375
MAE: 41.43922805786133
걸린시간: 3.8666458129882812

GPU

MSE: 2884.2001953125
MAE: 42.41807174682617
걸린시간: 3.645667552947998

...
...

StandardScaler

CPU

MSE: 3092.803955078125
MAE: 43.245643615722656
걸린시간: 3.824679374694824

GPU

MSE: 3053.258544921875
MAE: 43.16482925415039
걸린시간: 3.633452892303467

'''
