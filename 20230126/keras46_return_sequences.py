# 45_1 복붙

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
model = Sequential()                                    # (N, 3, 1)
model.add(LSTM(units=64, input_shape=(3, 1), 
               return_sequences=True))           # (N, 64) 
model.add(LSTM(32))                                                                                                                   
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

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
loss :  0.24280095100402832
[50, 60, 70]의 결과 :  [[74.288025]]

loss :  0.0497467927634716
[50, 60, 70]의 결과 :  [[76.53225]]


epochs=300 이후
loss :  0.040009837597608566
[50, 60, 70]의 결과 :  [[76.503174]]

loss :  0.0370771549642086
[50, 60, 70]의 결과 :  [[76.94191]]

loss :  0.042930059134960175
[50, 60, 70]의 결과 :  [[77.08521]]

loss :  0.022637562826275826
[50, 60, 70]의 결과 :  [[77.10377]]


epochs=500 이후
loss :  0.05146491527557373
[50, 60, 70]의 결과 :  [[76.74815]]

loss :  0.012704288586974144
[50, 60, 70]의 결과 :  [[77.960045]]

loss :  0.002558538457378745
[50, 60, 70]의 결과 :  [[78.26397]]




"""
             
             
             
             
             
             
             
             
             
             
             
                          