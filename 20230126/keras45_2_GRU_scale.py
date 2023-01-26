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

#2. 모델구성
model = Sequential()
model.add(GRU(units=10, input_shape=(3, 1)))                                      # 가장 많이 쓰이는 형태
                                                                                        
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

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
결과는???
loss :  0.17838144302368164
[50, 60, 70]의 결과 :  [[74.729385]]

loss :  0.10008583962917328
[50, 60, 70]의 결과 :  [[75.76024]]


epochs=500 이후
loss :  0.05929074063897133
[50, 60, 70]의 결과 :  [[76.46663]]

loss :  0.051339033991098404
[50, 60, 70]의 결과 :  [[76.60904]]

loss :  0.04151665419340134
[50, 60, 70]의 결과 :  [[77.25646]]


"""