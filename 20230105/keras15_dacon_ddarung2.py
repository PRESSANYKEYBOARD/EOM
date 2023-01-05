import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # 데이터에서 제외할 인덱스 칼럼이 0열에 있음. 열은 데이터가 아닌 인덱스라고 알려줘서 데이터로 들어가는 것을 방지.
# train_csv = pd.read_csv('./_data/ddarung/train.csv,' index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)   
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)  # (1459, 10) 카운트 분리하면 (1459, 9)
print(submission.shape) # (715, 1)  715개라는 평가 데이터를 알아야 되기 때문에, 삭제하면 안 된다.

print(train_csv.columns)
# ['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
    #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
    #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],

print(train_csv.info()) # 결측치 데이터가 2개가 있음. 임의의 데이터를 넣으면 오차가 커지기 때문에 제거해야 한다. 단, 데이터가 적을시에는 삭제하는 것이 오히려 치명적이다.
print(test_csv.info()) # 카운트 없음.
print(train_csv.describe())

#### 결측치 처리 1. 제거 ####
print(train_csv.isnull().sum()) # 중단점 클릭하고 F5 하면 중단점 직전까지 실행이 된다.

'''
결측치 데이터
hour                        0
hour_bef_temperature        2
hour_bef_precipitation      2
hour_bef_windspeed          9
hour_bef_humidity           2
hour_bef_visibility         2
hour_bef_ozone             76
hour_bef_pm10              90
hour_bef_pm2.5            117
count                       0
dtype: int64

'''

train_csv = train_csv.dropna()
print(train_csv.isnull().sum())     # 결측값 확인
print(train_csv.shape)  # (1328, 10)

x = train_csv.drop(['count'], axis=1)       # 작업할 때 axis가 1이면 행, 0이면 열
print(x)    # [1459 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)  # (1459, )

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=1234
)

print(x_train.shape, x_test.shape)  # (929, 9) (399, 9)
print(y_train.shape, y_test.shape)  # (929, ) (399, )

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(20))
model.add(Dense(125))
model.add(Dense(275))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))   # Dense 값을 임의로 조절해 주었다.
model.add(Dense(1))     # output

#3 컴파일, 훈련
import time     # 시간을 임포트
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])  # 가급적 유사지표에서는 mse
start = time.time()     # 시작 시간
model.fit(x_train, y_train, epochs=1000, batch_size=32)
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
rmse = RMSE(y_test, y_predict)
print("RMSE: ", rmse)

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

submission.to_csv(path + 'submission_01052322.csv')

'''
내가 찾은 답

result = pd.read_csv(path + 'submission.csv', index_col=0)
result['count'] = y_submit
result =.to_csv(path + 'submission_0105.csv', index = true)

'''

'''
하드웨어 사양 기준(노트북)

CPU: Intel Core i7-8750H  2.2Ghz (6C12T)
GPU: Nvidia Geforce GTX1060 6GB(notebook)
RAM: Samsung DDR4-3200 16GB x2 = 32GB

테스트 기준

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(20))
model.add(Dense(125))
model.add(Dense(275))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))   # Dense 값을 임의로 조절해 주었다.
model.add(Dense(1))     # output

model.fit(x_train, y_train, epochs=300, batch_size=32)

'''

# cpu 걸린시간: 97.43205142021179
# RMSE: 53.74518358208642

# gpu 걸린시간: 98.7128119468689
# RMSE: 53.8286581633613