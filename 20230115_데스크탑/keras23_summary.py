from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()                     # 모델에 대한 써머링이 나온다. / 아키텍처의 구조와 연산량을 보여준다.
                                    # output shape= ( , 노드의 갯수) / 노드의 갯수 말고도 바이어스도 같이 구한다.
                                    # 바이어스: 모든 레이어의 끝판왕
                                    # 바이어스 까지 되어있는 파라미터의 갯수
                                    

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 5)                 10

 dense_1 (Dense)             (None, 4)                 24

 dense_2 (Dense)             (None, 3)                 15

 dense_3 (Dense)             (None, 2)                 8

 dense_4 (Dense)             (None, 1)                 3

=================================================================
Total params: 60
Trainable params: 60
Non-trainable params: 0

"""