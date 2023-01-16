# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

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
x = datasets.data                                            # for training
y = datasets.target                                          # for predict

# scaler = MinMaxScaler()                                   # 정규화
scaler = StandardScaler()                                   # 표준화
scaler.fit(x)                                               # x의 값의 범위만큼 가중치 생성
x = scaler.transform(x)                                     # 생성된 가중치만큼의 data 형태 변형
print(x)
print(type(x))                                              # <class 'numpy.ndarray'>

# print("최소값:", np.min(x))                             # 0
# print("최대값:", np.max(x))                             # 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    random_state=333,
    test_size=0.2
    # stratify=y
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
model.add(Dense(5, activation='relu', input_shape=(13,)))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(120, activation='relu'))
model.add(Dense(30, activation='linear'))
model.add(Dense(1))

#3. 컴파일, 훈련
import time                                                                             # 시간을 임포트
model.compile(loss='mse', optimizer='adam',                # 다중 분류 손실함수를 categorical_crossentropy로 많이 사용한다. / 원 핫 인코딩을 거치게 된 후 사용.
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
CPU
0.7954449263407377
걸린시간 : 3.372718572616577

GPU
0.8346393459266429
걸린시간 : 7.357275724411011

'''