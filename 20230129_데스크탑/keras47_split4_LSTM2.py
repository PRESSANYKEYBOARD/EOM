# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))         
# 예상 y = 100, 107             

timesteps = 5                                                                   # x는 4개, y는 1개

def split_x(dataset, timesteps):
    aaa = [] # 빈 list 생성
    for i in range(len(dataset) - timesteps + 1):                               # for i in range(3->range(3): 0, 1, 2), range(4->2), range(5->1) : 반환하는 리스트 개수
        subset = dataset[i: (i+timesteps)]                                      # dataset[0(이상):3(미만)] [1:4] [2:5]: dataset 위치에 있는 값 반환
        aaa.append(subset) # append: 추가
    return np.array(aaa)

bbb = split_x(a, timesteps)
ccc = split_x(x_predict, timesteps-1)

print(ccc)
print(ccc.shape)

x = bbb[:, :-1] # 모든 행, 시작 ~ -1번째 열
y = bbb[:, -1] # 모든 행, -1번째 열(시작: 0번째 열)
x_predict = ccc[:,:]

print(x.shape, y.shape)

x = x.reshape(96,2,2)
x_predict = x_predict.reshape(7,2,2)
                                                                                # data feature가 홀수일 때는 reshape이 불가능하므로, 처음부터 data set을 짝수로 구비하기


'''
3차원 이상 작업 불가(split 후 reshape)
x_train, x_test, y_train, y_test = train_test_split()
x_train = x_train.reshape(7,4,1)
x_test = x_test.reshape(7,4,1)
x_predict = x_predict.reshape(7,4,1)

'''

# 실습
# LSTM 모델 구성

#2. 모델구성                                                            # 모델 구성 rnn = 2차원, rnn의 장기의존성을 해결하기 위해 LSTM이 탄생
model = Sequential()                                        
model.add(LSTM(units=64, input_shape=(2,2)))
                                                                        # reshape 시, timesteps*feature가 유지되도록 reshape                    # (N, 64)                                                                                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1))                                                     # # conv2D : input_dim 4 -> output_dim 4
                                                                        # LSTM: input_dim 3 -> output_dim 2 / # (N,3,1) -> (N, 64) 
                                                                        # return_sequences = True: (None, 3, 64) input_dim만 변화
                                                                        # ValueError: Input 0 of layer "lstm_1" is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 64)
                                                                        # 만일 return sequence 후 Dense 처리를 바로 할 경우, 총 3개의 차원 반환
                                                                    
#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_pred = np.array(x_predict).reshape(7,2,2)
                                                                        # (13, 3, 1) 형태와 동일하게 reshape로 해야 한다.
result = model.predict(y_pred)
print("Predict[100 ... 107]:\n ", result)

"""
LSTM
Predict[100 ... 107]:
  [[ 99.58168 ]
 [100.41204 ]
 [101.21953 ]
 [102.003525]
 [102.76354 ]
 [103.499306]
 [104.21057 ]]
 
LSTM2
Predict[100 ... 107]:
  [[ 99.299545]
 [100.0255  ]
 [100.72399 ]
 [101.39545 ]
 [102.04033 ]
 [102.659195]
 [103.25261 ]]

"""

