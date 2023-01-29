# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional      # Bidirectional 양방향으로!!!!!!!!!

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))         
# 예상 y = 100, 107             

timesteps = 5                                               # x는 4개, y는 1개

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):           
        subset = dataset[i : (i + timesteps)]               
        aaa.append(subset)
    return np.array(aaa)                                    # append: 한번씩 집어넣어라...

# 만들어랑
bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape)

x = bbb[:, :-1]
y = bbb[:, -1]
# print(x, y)
print(x.shape, y.shape)                                     # (96, 4) (96,)
x.reshape(96, 4, 1)

x_predict = split_x(x_predict, 4)                           # 106을 예측하기 위해...
print(x_predict.shape)                                      # (7, 4)
x_predict = x_predict.reshape(7, 4, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=1234
)

#2. 모델구성
model = Sequential()                               
# model.add(LSTM(units=64, input_shape=(4, 1), activation='relu'))
model.add(Bidirectional(LSTM(100), input_shape=(4, 1)))                                                     # Bidirection은 모델이 아니므로 모델 선택 필요
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=5)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = np.array(x_predict)             
                                                                        
result = model.predict(y_predict)
print('[100 ~ 107]의 결과:\n ',  result)

"""
loss :  0.4249722957611084
[100 ~ 107]의 결과:
  [[ 97.98216 ]
 [ 98.75017 ]
 [ 99.4938  ]
 [100.212524]
 [100.905914]
 [101.57367 ]
 [102.21566 ]]
 
""" 