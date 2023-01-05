import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    # 데이터에서 제외할 인덱스 칼럼이 0열에 있음. 열은 데이터가 아닌 인덱스라고 알려줘서 데이터로 들어가는 것을 방지.
# train_csv = pd.read_csv('./_data/ddarung/train.csv,' index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)  
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)  # (1459, 10) 카운트 분리하면 (1459, 9)

print(train_csv.columns)
# ['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
    #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
    #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],

print(train_csv.info()) # 결측치 데이터가 2개가 있음. 임의의 데이터를 넣으면 오차가 커지기 때문에 제거해야 한다. 단, 데이터가 적을시에는 삭제하는 것이 오히려 치명적이다.
print(test_csv.info()) # 카운트 없음.
print(train_csv.describe())

x = train_csv.drop(['count'], axis=1)
print(x)    # [1459 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)  # (1459, )

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=1234
)

print(x_train.shape, x_test.shape)  # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape)  # (1021, ) (438, )

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(50))
model.add(Dense(400))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])  # 가급적 유사지표에서는 mse
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss:', loss)

y_predict = model.predict(x_test)    # RMSE 출력을 위해 model.predict를 명시해준다.
print(y_predict)    # 실행하면 loss 값이 nan이라고 뜰 것인데, 뭔가 연산이 안 되고 있다는 뜻임. 없는 데이터(결측치)가 있었기 때문.

# 결측치 나쁜 놈!!!
# 결측치 때문에 To Be coutiune!!

'''
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)

# 제출할 놈
y_submit = model.predict(test_csv)

'''