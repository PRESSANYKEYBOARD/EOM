#47_4 복붙

import numpy as np
from tensorflow.keras.models import Sequential

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

# 3차원으로 늘릴꺼당.
x_train = x_train.reshape(72, 2, 2, 1)
x_test = x_test.reshape(24, 2, 2, 1)
x_predict = x_predict.reshape(7, 2, 2, 1)

print(x_train.shape, y_train.shape)                         # (72, 2, 2, 1) (72,)
print(x_test.shape, y_test.shape)                           # (24, 2, 2, 1) (24,)
print(x_predict.shape)                                      # (7, 2, 2, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#2. 모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='same', input_shape=(2, 2, 1))) 
model.add(Conv2D(64, (2,2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))           
# input_shape = (40000, )
                                                                                # (6만, 4만)이 인풋이야.
                                                                                # 6만 = batch_size, 4만 = input_dim
                                                                                
model.add(Dense(1, activation='softmax'))                                     

model.summary()

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

"""
