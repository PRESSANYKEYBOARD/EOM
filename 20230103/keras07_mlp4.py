import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10))
# print(range(10))
print(x.shape)  # (10,)
y=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
            [9,8,7,6,5,4,3,2,1,0]])
print(y.shape)  # (3,10)

y = y.T

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(55))
model.add(Dense(125))
model.add(Dense(38))
model.add(Dense(3)) # output

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs= 1800, batch_size=18)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss:', loss)

result = model.predict([9])
print('[9]의 예측값: ', result)

'''
결과: 9.97, 1.48, -0.29

'''

