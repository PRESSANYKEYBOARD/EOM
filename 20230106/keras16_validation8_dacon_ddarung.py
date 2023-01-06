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

train_csv = train_csv.dropna()
print(train_csv.isnull().sum())     # 결측값 확인
print(train_csv.shape)  # (1328, 10)

x = train_csv.drop(['count'], axis=1)       # 작업할 때 axis가 1이면 행, 0이면 열
print(x)    # [1459 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)  # (1459, )

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=1234
)

x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test,
    train_size=0.7, shuffle=True, random_state=1234
)

print(x_train.shape, x_test.shape)  # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape)  # (1021, ) (438, )

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=9))
model.add(Dense(40, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
import time     # 시간을 임포트
model.compile(loss='mse', optimizer='adam')     # 가급적 유사지표에서는 mse           
start = time.time()     # 시작 시간
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25)        # val_loss 즉, 검증할 때 손실값이 출력된다.
                                        # 기준을 잡을 때, val_loss로 기준을 잡는다.   
end = time.time()       # 종료 시간

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss:', loss)

y_predict = model.predict(x_test)    # RMSE 출력을 위해 model.predict를 명시해준다.
print(y_predict)    # 실행하면 loss 값이 nan이라고 뜰 것인데, 뭔가 연산이 안 되고 있다는 뜻임. 없는 데이터(결측치)가 있었기 때문.

# 결측치 나쁜 놈!!!
# 결측치 때문에 To Be coutiune!!

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)

print("걸린시간 :", end - start)

# 제출할 놈
y_submit = model.predict(test_csv)
# print(y_submit)       # nan이 출력되는 걸로 봐선 test_csv에도 뭔가 결측치가 있었다는 뜻임.
# print(y_submit.shape)   # (715, 1)

# .to_csv를 사용해서
# submission_0105.csv를 완성하시오!!!

# print(submission)
submission['count'] = y_submit      # 대여수를 예측한 카운트에 submission에 넣어준다.
# print(submission)

submission.to_csv(path + 'submission_01062325.csv')

'''
값 조정

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=1234
)

x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test,
    train_size=0.7, shuffle=True, random_state=1234
)

model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25)

CPU 기준          
RMSE:  54.89655169518985
R2:  0.592370876546277
val_loss: 2751.8584
걸린시간: 70.8092532157898

GPU 기준          
RMSE:  54.139213840251145
R2:  0.603540376143576
val_loss: 2931.0547
걸린시간: 132.62327790260315

'''