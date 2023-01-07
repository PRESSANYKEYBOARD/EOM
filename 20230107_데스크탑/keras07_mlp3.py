# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np                                          # 넘파이를 임포트시키고 간단하게 np로 명시해준다.
from tensorflow.keras.models import Sequential              # Sequential은 레이어에 순차적으로 연산
from tensorflow.keras.layers import Dense                   # Dense는 완전 연결층을 구현하는 레이어 모델

#1. 데이터
x = np.array([range(10), range(21, 31), range(201,211)])    # range(0, 10) 0~9까지. 0부터 10-1까지이다. # 인덱스는 번호가 1이 아닌 0부터 시작하기 때문이다. 그냥 외우자...
                                                            # 0부터 10개의 수 = 10개의 데이터 / 21~30 / 201~210, 마지막 수-1
print(x.shape)                                              # (3, 10) → (행, 렬)

y=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
print(y.shape)                                              # (2, 10)

x = x.T                                                     # 전치하다. 즉, x의 행과 열을 바꾼다 # 열 = column, 피처, 특성, 컬럼 다 같은 말이니 꼭 숙지하자!!!
y = y.T                                                     # y도 전치해서 (10, 2)

model=Sequential()                                          # 모델은 순차적으로 구성하겠다.
model.add(Dense(3, input_dim=3))                            # 3열이라서 iuput_dim은 3       # x열의 개수
model.add(Dense(55))                                
model.add(Dense(30))                                        # Dense값을 임의로 조절
model.add(Dense(18))                                 
model.add(Dense(2))                                         # output=2                     # y열의 개수

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')                 # loss값을 최소화 위해 mae를 쓰겠다는 뜻. loss를 최적화 하기 위해 adam을 사용하는데, 평타 이상은 친다.
model.fit(x, y, epochs=100, batch_size=32)                  # epoch: 훈련을 여러번 시키는것. fit: 훈련시킴. # epoch 너무 많이하면 오히려 성능이 떨어지는 경우가 있기 때문에, 적절한 epoch 수치를 찾아내야 함. # 이번엔 훈련 수치를 100번으로 조정.
                                                            # batch_size가 크면 속도가 빨라지고 정확도가 떨어짐. 과적합(overfitting)을 고려해서 조절해야 함. # batch size default=32(기본값) # 여기서는 기본값으로 임의로 세팅

#4. 평가, 예측
loss = model.evaluate(x, y)                                   # loss 값으로 평가, 반환 # evaluate : 들어가는 데이터는 훈련데이터가 들어가면 안된다. # loss 수치가 낮으면 가중치에 최적화. 판단의 기준은 predict보다 loss(0과 가까울수록 좋다)이다.
                                                            # evaluate 에도 배치 사이즈가 존재. defalut 값 = 32
print('loss : ', loss)

result = model.predict([[9, 30, 210]])
print("[[9, 30, 210]]의 예측 결과 : ", result)

'''
[[9, 30, 210]]의 결과(1트): 11.954778   0.8934535,  loss값: 1.355769157409668
[[9, 30, 210]]의 결과(2트): 10.202575   1.2951858,  loss값: 0.7003161311149597
[[9, 30, 210]]의 결과(3트): 6.037028  1.4467666,  loss값: 1.404431939125061
[[9, 30, 210]]의 결과(4트): 9.592003  0.9397922,  loss값: 0.4598553776741028
[[9, 30, 210]]의 결과(5트): 7.405886  1.6344068,  loss값: 0.6859695911407471

'''