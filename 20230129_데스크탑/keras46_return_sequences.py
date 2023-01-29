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
x = np.array([[1,2,3],[2,3,4],[3,4,5],
              [4,5,6],[5,6,7],[6,7,8],
              [7,8,9],[8,9,10],[9,10,11],
              [10,11,12],[20,30,40],
              [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predit = np.array([50,60,70])                     # I wanna 80?


print(x.shape, y.shape)                             # (13, 3) (13,)
x = x.reshape(13, 3, 1) 

#2. 모델구성                                                            # 모델 구성 rnn = 2차원, rnn의 장기의존성을 해결하기 위해 LSTM이 탄생
model = Sequential()                                    # (N, 3, 1)
model.add(LSTM(units=64, input_shape=(3, 1), 
               return_sequences=True))                  # (N, 64) 
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
model.fit(x, y, epochs=500, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_pred = np.array([50, 60, 70]).reshape(1, 3, 1)                        # (3, ) 에러가 난다.
                                                                        # (13, 3, 1) 형태와 동일하게 reshape로 해야 한다.
result = model.predict(y_pred)
print('[50, 60, 70]의 결과 : ',  result)   

"""
loss :  0.012367433868348598
[50, 60, 70]의 결과 :  [[77.71488]]

"""