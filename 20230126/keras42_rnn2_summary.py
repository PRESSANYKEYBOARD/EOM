# 40 복붙

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

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
print(x.shape)                                                      # (7, 3, 1)

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(64, activation='relu', input_shape=(3, 1)))                         # RNN은 이렇게 명시를 해줘야 한다. / RNN은 3차원인데 행 무시를 하면 2차원
                                                                                        # DNN은 2차원 이상, 차원을 올릴때마다 인풋쉐이프 해주면 됨
                                                                                        # CNN은 4차원
                                                                                        
                                                                                        # (N, 3, 1) → (batch, timesteps, feature)
                                                                                        
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1))                   

model.summary() 

# SimpleRNN이 왜 파라미터 개수가 4224?
# of params = (Dh * Dh) + (Dh * d) + (Dh)
#  = (64 * 64) + (64 * 1) + (64) 
#  = 4224
                                                                                        
# 파라미터 아웃값 * (파라미터 아웃값 + 디멘션 값 + 1(바이어스))
# 64 * (64+1+1) = 
# units * ( feature units + bias + units )

  



"""
loss :  0.942895770072937
[8, 9, 10]의 결과 :  [[8.292752]]


레이어 늘리고, epochs=200, batch_size=7의 결과
loss :  2.6832454750547186e-05
[8, 9, 10]의 결과 :  [[10.999339]]

loss :  0.0033894218504428864
[8, 9, 10]의 결과 :  [[11.155454]]

loss :  7.487023685825989e-05
[8, 9, 10]의 결과 :  [[11.044466]]

loss :  9.855650205281563e-06
[8, 9, 10]의 결과 :  [[11.027463]]

loss :  4.8185709601966664e-05
[8, 9, 10]의 결과 :  [[11.005507]]

.
.
.

epochs=300
loss :  1.5797157004726614e-07
[8, 9, 10]의 결과 :  [[10.999709]]

loss :  7.670921036151412e-07
[8, 9, 10]의 결과 :  [[11.007254]]

"""  