# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])                          # 데이터 / 여기서는 주가 데이터라고 치자.
                                                                    # 데이터 형태 (10, )
# y = ???                                                               

x = np.array([[1,2,3], 
              [2,3,4], 
              [3,4,5], 
              [4,5,6], 
              [5,6,7], 
              [6,7,8], 
              [7,8,9]])                                             # RNN에서 쓸 쉬 있게 데이터를 3일치씩 잘랐다.

y = np.array([4, 5, 6, 7, 8, 9, 10])                                # 8, 9 다음에 뭐?

print(x.shape, y.shape)                                             # (7, 3) (7,)

x = x.reshape(7, 3, 1)                                              # 1개씩 연산을 해줬다는 걸 명시하기 위해 reshape를 해준다.
                                                                    # → [[[1],[2],[3]], 
                                                                    #    [[2],[3],[4]], ...]
                                                                    
'''
# reshape을 해주는 이유: 추후 1개씩, 2개씩, 3개씩 연산이 필요한 경우가 생기므로
input_dim=1인 경우 생략 가능
x = np.array([[[1],[2],[3]], 
              [[2],[3],[4]], 
              [[3],[4],[5]], 
              [[4],[5],[6]], 
              [[5],[6],[7]], 
              [[6],[7],[8]], 
              [[7],[8],[9]]]) # (7,3,1)
# reshape이 맞는지 확인하는 법
# Data 개수 제외 다른 차원의 개수 모두 곱해서 동일한지 확인

'''
                                                                    
print(x.shape)                                                      # (7, 3, 1)

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(64, activation='relu', input_shape=(3, 1)))                         # RNN은 이렇게 명시를 해줘야 한다. / RNN은 3차원인데 행 무시를 하면 2차원
                                                                                        # DNN은 2차원 이상, 차원을 올릴때마다 인풋쉐이프 해주면 됨
                                                                                        # CNN은 4차원
                                                                                        # input_dim=1: 1->2, 2->3, 3->4 과정을 거침
                                                                                        
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1))                   

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=7)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_pred = np.array([8, 9, 10]).reshape(1, 3, 1)                      # (3, ) 에러가 난다.
                                                                    # (7, 3, 1) 형태와 동일하게 reshape로 해야 한다.
                                                                    # DNN layer는 2차원 이상 -> reshape 필요
                                                            
                                                                    # training. (7, 3, 1)
                                                                    # training data와 predict data의 shape이 다름
result = model.predict(y_pred)
print('[8, 9, 10]의 결과 : ',  result)

"""
loss :  1.8677061234484427e-05
[8, 9, 10]의 결과 :  [[11.049061]]

"""