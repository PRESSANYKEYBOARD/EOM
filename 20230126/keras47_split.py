import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

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

# 실습
# LSTM 모델 구성

#2. 모델구성
model = Sequential()                               
model.add(LSTM(units=64, input_shape=(4, 1), 
               return_sequences=True))          
model.add(LSTM(32))                                                                                                                   
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

y_pred = np.array([7, 8, 9, 10]).reshape(1, 4, 1)                      
                                                                        
result = model.predict(y_pred)
print('[7, 8, 9, 10]의 결과 : ',  result)   


"""
결과는???
loss :  0.006226222962141037
[7, 8, 9, 10]의 결과 :  [[10.869601]]

loss :  0.0006145701627247036
[7, 8, 9, 10]의 결과 :  [[10.9713545]]


epochs=500 이후
loss :  6.0656428104266524e-05
[7, 8, 9, 10]의 결과 :  [[10.822388]]

loss :  0.0013617351651191711
[7, 8, 9, 10]의 결과 :  [[11.0485325]]

"""






