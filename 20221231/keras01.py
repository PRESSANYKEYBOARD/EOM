import tensorflow as tf
#텐서폴드를 임포트하지만 as로 줄여서 tf라고 칭한다.
print(tf.__version__)

import numpy as np

#1. 데이터 입력
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성 입력
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(1,input_dim=1))

#3. 컴퍼일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=10)

#4. 평가, 예측
result=model.predict([4])
print('결과:',result)