# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM

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
# model.add(SimpleRNN(64, activation='relu', input_shape=(3, 1)))                         # RNN은 이렇게 명시를 해줘야 한다. / RNN은 3차원인데 행 무시를 하면 2차원
                                                                                        # DNN은 2차원 이상, 차원을 올릴때마다 인풋쉐이프 해주면 됨
                                                                                        # CNN은 4차원
                                                                                        # input_dim=1: 1->2, 2->3, 3->4 과정을 거침
                                                                                        
#                                                                                         # (N, 3, 1) → (batch, timesteps, feature)

# model.add(SimpleRNN(units=64, input_length=3, input_dim=1))                                 # 이렇게 똑같이 쓸 수 있다.
# model.add(SimpleRNN(units=64, input_dim=1, input_length=3))                               # 거꾸로도 가능해요~ 

# model.add(LSTM(units=64, input_length=3, input_dim=1))
model.add(LSTM(units=10, input_shape=(3, 1)))                                             # 가장 많이 쓰이는 형태                                                                                       
                                                                                        
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

# 심플
# 10 * (10 + 1 + 1) = 120

# LSTM
# Param = 4*((input_shape_size +1) * ouput_node + output_node^2)
# 4 * (( 1 + 1 ) * 10 + 100)
# 4 * (2 * 10 + 100)
# 4 * 120 = 480              

# # 현지쌤 소스 복붙 
# # 2. Model Construction
# model = Sequential()
# model.add(SimpleRNN(units=64, input_length=3, input_dim=1))
# model.add(SimpleRNN(units=64, input_shape=(3,1))) # input_length=3, input_dim=1, input_dim 단위로 연산
# # (N, 3, 1) = ([batch(데이터 개수), timesteps, feature])
# # batch: train data 총 set 수
# # timesteps: 1set의 train data 수
# # feature: train data 개수를 한 번에 몇 개씩 계산
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))
# model.summary()

'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 simple_rnn (SimpleRNN)      (None, 64)                4224
 dense (Dense)               (None, 32)                2080
 dropout (Dropout)           (None, 32)                0
 dense_1 (Dense)             (None, 32)                1056
 dropout_1 (Dropout)         (None, 32)                0
 dense_2 (Dense)             (None, 16)                528
 dense_3 (Dense)             (None, 1)                 17
=================================================================
Total params: 7,905
Trainable params: 7,905
Non-trainable params: 0
_________________________________________________________________
1번째 Param #
Total params = recurrent_weights + input_weights + biases
= (units*units)+(features*units) + (1*units)
= units(units + feature + 1)

'''

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 10)                480

 dense (Dense)               (None, 256)               2816


 dense_2 (Dense)             (None, 32)                4128

 dense_3 (Dense)             (None, 1)                 33

=================================================================
Total params: 40,353
Trainable params: 40,353
Non-trainable params: 0

"""

