# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np                                  # 넘파이를 임포트시키고 간단하게 np로 명시해준다.
import tensorflow as tf                             # 텐서플로를 임포트 시킨다. 하지만 텐서플로를 그대로 쓰기에는 이름이 너무 기니, tf로 간단하게 명시해준다.
from tensorflow.keras.models import Sequential      # Sequential은 레이어에 순차적으로 연산
from tensorflow.keras.layers import Dense           # Dense는 완전 연결층을 구현하는 레이어 모델

#1. 데이터
x = np.array([1,2,3,4,5,6])                         # 넘파이 형식의 어레이 데이터
y = np.array([1,2,3,5,4,6])                         # 넘파이 형식의 어레이 데이터

model=Sequential()                                  # 모델은 순차적으로 구성하겠다.
model.add(Dense(3, input_dim=1))                    # y줄의(output) 123이 1, x줄의(input) 123이 dim=1의 1. # Dense=(y=yx+b)를 1번 계산. dim = dimention의 약자 
model.add(Dense(55))                                
model.add(Dense(30))                                # Dense값을 임의로 조절
model.add(Dense(18))                                 
model.add(Dense(1))                                 # output=1

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')         # loss값을 최소화 위해 mae를 쓰겠다는 뜻. loss를 최적화 하기 위해 adam을 사용하는데, 평타 이상은 친다.
model.fit(x, y, epochs=50, batch_size=5)            # epoch: 훈련을 여러번 시키는것. fit: 훈련시킴. # epoch 너무 많이하면 오히려 성능이 떨어지는 경우가 있기 때문에, 적절한 epoch 수치를 찾아내야 함. # 이번엔 훈련 수치를 50번으로 조정.
                                                    # batch_size: 6개를 2개 단위로 나누어서 훈련(batch), batch_size가 크면 속도가 빨라지고 정확도가 떨어짐. 과적합(overfitting)을 고려해서 조절해야 함. # batch size default=32(기본값)
                                                    # 하이퍼 파라미터 튜닝(3)

#4. 평가, 예측
loss = model.evaluate(x, y)                         # loss 값으로 평가, 반환 # evaluate : 들어가는 데이터는 훈련데이터가 들어가면 안된다. # loss 수치가 낮으면 가중치에 최적화. 판단의 기준은 predict보다 loss(0과 가까울수록 좋다)이다.
print('loss : ', loss)
results = model.predict([6])
print('6의 결과: ' ,results)

'''
6의 결과(1트): -0.6319059,  loss값: 3.8478546142578125
6의 결과(2트): 4.967203,    loss값: 0.6112203598022461
6의 결과(3트): 6.1049566,   loss값: 0.4168446362018585
6의 결과(4트): 5.9812546,   loss값: 0.33768579363822937
6의 결과(5트): 5.8797007,  loss값: 0.3619554936885834

'''