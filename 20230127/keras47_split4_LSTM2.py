#47_2 복붙

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))         
# 예상 y = 100, 107             

timesteps = 5                                               # x는 4개, y는 1개

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):           
        subset = dataset[i : (i + timesteps)]               
        aaa.append(subset)
    return np.array(aaa)                                    # append: 한번씩 집어넣어라...

# 만들어랑
bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape)

x = bbb[:, :-1]
y = bbb[:, -1]
# print(x, y)
print(x.shape, y.shape)                                     # (96, 4) (96,)

x_predict = split_x(x_predict, 4)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=1234
)

print(x_train.shape, y_train.shape)                         # (72, 4) (72,)
print(x_test.shape, y_test.shape)                           # (24, 4) (24,)

# 피쳐를 2로 바꿀꺼야
x_train = x_train.reshape(72, 2, 2)
x_test = x_test.reshape(24, 2, 2)
x_predict = x_predict.reshape(7, 2, 2)

print(x_train.shape, y_train.shape)                         # (72, 2, 2) (72,)
print(x_test.shape, y_test.shape)                           # (24, 2, 2) (24,)
print(x_predict.shape)                                      # (7, 2, 2)

#2. 모델구성
model = Sequential()                               
model.add(LSTM(units=64, input_shape=(2, 2)))        
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=5)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = np.array(x_predict)             
                                                                        
result = model.predict(y_predict)
print('[100 ~ 107]의 결과:\n ',  result) 


"""
결과는???
loss :  0.0028117254842072725
[100 ~ 107]의 결과:
  [[100.17116 ]
 [101.185974]
 [102.20137 ]
 [103.217415]
 [104.23409 ]
 [105.25135 ]
 [106.26932 ]]

loss :  0.0002512639621272683
[100 ~ 107]의 결과:
  [[100.00816 ]
 [101.00677 ]
 [102.00518 ]
 [103.00343 ]
 [104.001495]
 [104.99938 ]
 [105.997086]]
 
 loss :  0.00039378413930535316
[100 ~ 107]의 결과:
  [[100.03664 ]
 [101.0397  ]
 [102.04283 ]
 [103.046036]
 [104.04927 ]
 [105.05256 ]
 [106.055885]]
 
"""

"""
loss :  0.30883926153182983
[100 ~ 107]의 결과:
  [[ 98.82868 ]
 [ 99.673164]
 [100.49849 ]
 [101.30475 ]
 [102.09211 ]
 [102.86069 ]
 [103.61067 ]]

"""
