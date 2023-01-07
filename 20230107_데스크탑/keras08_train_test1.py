# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np                                          # 넘파이를 임포트시키고 간단하게 np로 명시해준다.
from tensorflow.keras.models import Sequential              # Sequential은 레이어에 순차적으로 연산
from tensorflow.keras.layers import Dense                   # Dense는 완전 연결층을 구현하는 레이어 모델

'''
전체데이터를 훈련시키면 과적합(overfit)이 발생하니, 훈련 데이터(train set)와 평가 데이터(test set)로 분리해서 관리하는게 적합하다. 
ex) 7:3  |-------|---|

'''

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])                      # (10, )
# y = np.array(range(10))                                   # (10, )

x_train = np.array([1,2,3,4,5,6,7])                         # (7, )     # train과 test 행이 달라도 열이 같으면 괜찮음.
x_test = np.array([8,9,10])                                 # (3, )     # x, y를 train과 test로 분리할 때 %는 동일하게 조정.
y_train = np.array(range(7))
y_test = np.array(range(7, 10))

#2. 모델구성
model=Sequential()                                          # 모델은 순차적으로 구성하겠다.
model.add(Dense(3, input_dim=1))                            # 1열이라서 iuput_dim은 1       # x열의 개수
model.add(Dense(55))                                
model.add(Dense(100))                                       # Dense값을 임의로 조절
model.add(Dense(80))
model.add(Dense(60)) 
model.add(Dense(25))                                  
model.add(Dense(1))                                         # output=1                     # y열의 개수

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')                 # loss값을 최소화 위해 mae를 쓰겠다는 뜻. loss를 최적화 하기 위해 adam을 사용하는데, 평타 이상은 친다.
model.fit(x_train, y_train, epochs=100, batch_size=32)      # epoch: 훈련을 여러번 시키는것. fit: 훈련시킴. # epoch 너무 많이하면 오히려 성능이 떨어지는 경우가 있기 때문에, 적절한 epoch 수치를 찾아내야 함. # 이번엔 훈련 수치를 100번으로 조정.
                                                            # batch_size가 크면 속도가 빨라지고 정확도가 떨어짐. 과적합(overfitting)을 고려해서 조절해야 함. # batch size default=32(기본값) # 여기서는 기본값으로 임의로 세팅

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                         # loss 값으로 평가, 반환 # evaluate : 들어가는 데이터는 훈련데이터가 들어가면 안된다. # loss 수치가 낮으면 가중치에 최적화. 판단의 기준은 predict보다 loss(0과 가까울수록 좋다)이다.
                                                            # evaluate 에도 배치 사이즈가 존재. defalut 값 = 32
print('loss : ', loss)

result = model.predict([11])                                   
print("[11]의 예측 결과 : ", result)


'''
[11]의 결과(1트): 10.529818,  loss값: 0.4378634989261627
[11]의 결과(2트): 10.139906,  loss값: 0.1221253052353859
[11]의 결과(3트): 10.254752,  loss값: 0.213580921292305
[11]의 결과(4트): 10.485712,  loss값: 0.39686569571495056
[11]의 결과(5트): 9.4385195, loss값: 0.4581092298030853
'''