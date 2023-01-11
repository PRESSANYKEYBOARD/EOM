import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

#1.  데이터
datasets = load_breast_cancer()
# print(datasets)                         # x 값은 많은 데이터, y 값은 0과 1로 구성되어 있음.
#                                         # target_names: 암에 걸렸다 / 안 걸렸다.
#                                         # 


# print(datasets.DESCR)
# print(datasets.feature_names)           # 30개

'''
    radius (mean):                        6.981  28.11
    texture (mean):                       9.71   39.28
    perimeter (mean):                     43.79  188.5
    area (mean):                          143.5  2501.0
    smoothness (mean):                    0.053  0.163
    compactness (mean):                   0.019  0.345
    concavity (mean):                     0.0    0.427
    concave points (mean):                0.0    0.201
    symmetry (mean):                      0.106  0.304
    fractal dimension (mean):             0.05   0.097
    radius (standard error):              0.112  2.873
    texture (standard error):             0.36   4.885
    perimeter (standard error):           0.757  21.98
    area (standard error):                6.802  542.2
    smoothness (standard error):          0.002  0.031
    compactness (standard error):         0.002  0.135
    concavity (standard error):           0.0    0.396
    concave points (standard error):      0.0    0.053
    symmetry (standard error):            0.008  0.079
    fractal dimension (standard error):   0.001  0.03
    radius (worst):                       7.93   36.04
    texture (worst):                      12.02  49.54
    perimeter (worst):                    50.41  251.2
    area (worst):                         185.2  4254.0
    smoothness (worst):                   0.071  0.223
    compactness (worst):                  0.027  1.058
    concavity (worst):                    0.0    1.252
    concave points (worst):               0.0    0.291
    symmetry (worst):                     0.156  0.664
    fractal dimension (worst):            0.055  0.208

'''
                                
x = datasets['data']                  # datasets.data와 같음
y = datasets['target']                # datasets.target 같음
# print(x.shape, y.shape)               # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, test_size=0.2 
)                                                           # train은 대략 404개 정도

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)                                           # x_train에 대한 범위의 가중치 생성
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x_train = scaler.fit_transform(x_train)                     # 한 줄로 정리

#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='linear', input_shape=(30,)))            # input_dim=30
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))                               # 0과 1 사이의 값을 찾아야 되기 때문에 sigmoid로 맞춰준다.
                                                                        # 2진 분류네???

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

#4. 평가 ,예측
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

MSE: 0.03159179165959358
MAE: 0.047080107033252716
걸린시간: 3.9775664806365967

GPU

MSE: 0.0359627939760685
MAE: 0.05540589243173599
걸린시간: 4.064855098724365

...
...

StandardScaler

CPU

MSE: 0.028578316792845726
MAE: 0.036949750036001205
걸린시간: 3.964484214782715

GPU

MSE: 0.041600387543439865
MAE: 0.047017332166433334
걸린시간: 4.105473279953003

'''