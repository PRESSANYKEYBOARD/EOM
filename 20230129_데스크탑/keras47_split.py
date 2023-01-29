# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU

#1. 데이터
a = np.array(range(1, 11))                                   
timesteps = 5                                               # n번씩 자르겠다.

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):           # 5-3+1 만큼 반복해! / (0 ,1, 2)
        subset = dataset[i : (i + timesteps)]               # a[0 : 3] = [1, 2, 3]
        aaa.append(subset)
    return np.array(aaa)                                    # append: 한번씩 집어넣어라...

bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape)                                     # (6, 4) (6,)

x_predict = np.array([7, 8, 9, 10])                         # 11이 나오도록 예측해보자...

x = x.reshape(6, 4, 1)

# 실습
# LSTM 모델 구성

#2. 모델구성                                                 # 모델 구성 rnn = 2차원, rnn의 장기의존성을 해결하기 위해 LSTM이 탄생
model = Sequential()                                        
model.add(LSTM(units=64, input_shape=(4, 1), 
               return_sequences=True))                      # (N, 64) 
model.add(LSTM(32))                                                                                                                   
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1))                                                     # # conv2D : input_dim 4 -> output_dim 4
                                                                        # LSTM: input_dim 3 -> output_dim 2 / # (N,3,1) -> (N, 64) 
                                                                        # return_sequences = True: (None, 3, 64) input_dim만 변화
                                                                        # ValueError: Input 0 of layer "lstm_1" is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 64)
                                                                        # 만일 return sequence 후 Dense 처리를 바로 할 경우, 총 3개의 차원 반환
                                                                    
#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_pred = np.array([7, 8, 9, 10]).reshape(1, 4, 1)                        # (3, ) 에러가 난다.
                                                                        # (13, 3, 1) 형태와 동일하게 reshape로 해야 한다.
result = model.predict(y_pred)
print('[7, 8, 9, 10]의 결과 : ',  result)

"""
loss :  2.1818448658450507e-05
[7, 8, 9, 10]의 결과 :  [[10.937407]]


"""

