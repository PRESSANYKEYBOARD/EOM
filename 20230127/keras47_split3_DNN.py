import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

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

#2. 모델구성
model = Sequential()                               
model.add(Dense(units=64, input_shape=(4, ), activation='relu'))
model.add(Dropout(0.3))       
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=5)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = np.array(x_predict)             
                                                                        
result = model.predict(y_predict)
print('[100 ~ 107]의 결과:\n ',  result)     
                                                                
"""
결과는???
loss :  0.0003321784606669098

"""