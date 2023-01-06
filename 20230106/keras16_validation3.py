import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x, y,       # 먼저, x와 y의 데이터를 받아서 0.65로 자른다.
    train_size=0.65, shuffle=False                              # 값이 순차적으로 나와야 되기 때문에 shuffle을 False로 설정.
)

x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test,   # 그 다음, x_test와 y_test의 데이터를 받아서 0.5의 비율로 동일하게 자른다.
    train_size=0.5, shuffle=False                               # 값이 순차적으로 나와야 되기 때문에 shuffle을 False로 설정.
)

'''
역시 모르겠으면 프린트로 출력하는 값을 보고 조정해주면 된다.

print(x_train)
print(x_test)
print(x_validation)
print(y_train)
print(y_test)
print(y_validation)

'''

# [실습] 잘라봐!!!
# train_test_split로 잘라봐!!!
# 10:3:3으로 나눠라

# x_train = x[0:10]          # x[:10]    # x[:-6]     동일한 값들이 나온다.
# x_test = x[10:13]          # x[-6:-3] 
# x_validation = x[13:16]    # x[13:]    # x[-3:]
# y_train = y[0:10]          # y[:10]    # y[:-6]
# y_test = y[10:13]          # y[-6:-3] 
# y_validation = y[13:16]    # y[13:]    # y[-3:]

# x_train = np.array(range(1, 11))        # 훈련 데이터
# y_train = np.array(range(1, 11))        
# x_test = np.array([11,12,13])           # 평가 데이터
# y_test = np.array([11,12,13])
# x_validation = np.array([14,15,16])     # 검증 데이터(문제지를 푼다.)
# y_validation = np.array([14,15,16])     

'''
#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_validation, y_validation))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

result = model.predict([17])
print("17의 예측값 :", result)
'''