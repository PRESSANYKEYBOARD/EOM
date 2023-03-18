# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import tensorflow as tf                             # 텐서플로를 임포트 시킨다. 하지만 텐서플로를 그대로 쓰기에는 이름이 너무 기니, tf로 간단하게 명시해준다.
print(tf.__version__)                               # 현재 텐서플로의 버전을 출력한다. 2.7.4
import numpy as np                                  # 1번 라인과 마찬가지로 넘파이를 임포트시키고 간단하게 np로 명시해준다.

#1. 데이터 입력
x = np.array([1,2,3,4,5,6,7,8,9,10])                # 넘파이 어레이 형식의 데이터
y = np.array([1,2,3,4,5,6,7,8,9,10])                # 넘파이 어레이 형식의 데이터

# [13] 예측해보자... 끼에에엨...

#2. 모델구성
from tensorflow.keras.models import Sequential      # Sequential은 레이어에 순차적으로 연산
from tensorflow.keras.layers import Dense           # Dense는 완전 연결층을 구현하는 레이어 모델

model=Sequential()                                  # 모델은 순차적으로 구성하겠다.
model.add(Dense(1, input_dim=1))                    # y줄의(output) 123이 1, x줄의(input) 123이 dim=1의 1. # Dense=(y=yx+b)를 1번 계산. dim = dimention의 약자

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')         # loss값을 최소화 위해 mae를 쓰겠다는 뜻. loss를 최적화 하기 위해 adam을 사용하는데, 평타 이상은 친다.
model.fit(x, y, epochs=10)                          # epoch: 훈련을 여러번 시키는것. fit: 훈련시킴. # epoch 너무 많이하면 오히려 성능이 떨어지는 경우가 있기 때문에, 적절한 epoch 수치를 찾아내야 함.

#4. 평가, 예측
result = model.predict([13])                        # 13에 대한 예측값의 결과는???
print('결과:', result)


'''
결과(1트): -9.252584 
결과(2트): -2.1215258
결과(3트): 12.553874
결과(4트): 19.823421

'''