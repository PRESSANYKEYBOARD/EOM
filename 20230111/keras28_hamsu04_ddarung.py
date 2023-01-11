import numpy as np
from tensorflow.keras.models import Sequential, Model                           # model은 인풋 레이어를 명시해줘야함.
from tensorflow.keras.layers import Dense, Input
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

#1. 데이터
# path = './_data/ddarung/'
# path = '../_data/ddarung/'
path = 'C:/study/_data/ddarung/'                            # 절대 경로
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    # 데이터에서 제외할 인덱스 칼럼이 0열에 있음. 열은 데이터가 아닌 인덱스라고 알려줘서 데이터로 들어가는 것을 방지.
                                                            # train_csv = pd.read_csv('./_data/ddarung/train.csv,' index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)  
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)                                      # (1459, 10) 카운트 분리하면 (1459, 9)

print(train_csv.columns)
                                                            # ['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                                                                #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                                                                #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],

print(train_csv.info())                                     # 결측치 데이터가 2개가 있음. 임의의 데이터를 넣으면 오차가 커지기 때문에 제거해야 한다. 단, 데이터가 적을시에는 삭제하는 것이 오히려 치명적이다.
print(test_csv.info())                                      # 카운트 없음.
print(train_csv.describe())

train_csv = train_csv.dropna()
print(train_csv.isnull().sum())                             # 결측값 확인
print(train_csv.shape)                                      # (1328, 10)

x = train_csv.drop(['count'], axis=1)                       # 작업할 때 axis가 1이면 행, 0이면 열
print(x)                                                    # [1459 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)                                              # (1459, )
                                                           
x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, test_size=0.2 
)                                                               # train은 대략 1167개 정도

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)                                             # x_train에 대한 범위의 가중치 생성
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
x_train = scaler.fit_transform(x_train)                         # 한 줄로 정리
x_test = scaler.transform(x_test)                               
test_csv = scaler.transform(test_csv)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(5, input_shape=(9, )))                          # 스칼라가 9개로 볼 수 있기 때문에, (9, ) 라고 해석할 수 있다.
#                                                                 # (100, 10, 5) 라는 데이터가 들어가면, 행 무시 열 우선이기 때문에 (10, 5)로 들어감.
#                                                                 # 앞으로는 이렇게 쓰자!!!
# model.add(Dense(40, activation='sigmoid'))
# model.add(Dense(120, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='linear'))

# 2. 모델구성(함수형)
input1 = Input(shape=(9, ))                                     
Dense1 = Dense(50, activation='relu')(input1)
Dense2 = Dense(50, activation='relu')(Dense1)
Dense3 = Dense(64, activation='sigmoid')(Dense2)
Dense4 = Dense(32, activation='relu')(Dense3)
Dense5 = Dense(16, activation='linear')(Dense4)
output1 = Dense(1, activation='linear')(Dense5)
model = Model(inputs=input1, outputs=output1)                    # 시작하는 부분과 끝 부분이 어디인지 알려주는 부분
model.summary()

#3. 컴파일, 훈련
import time                                                                             # 시간을 임포트
model.compile(loss='mse', optimizer='adam',                
             metrics=['mae'])
                         
start = time.time()                                                                     # 시작 시간
model.fit(x_train, y_train, epochs=200, batch_size=32,
          validation_split=0.2,
          verbose=1)                                                  # val_loss 즉, 검증할 때 손실값이 출력된다.
                                                                      # 기준을 잡을 때, val_loss로 기준을 잡는다.   
end = time.time()                                                     # 종료 시간

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse)
print('mae:', mae)

y_predict = model.predict(x_test)

print("y_test(원래값):", y_test)
r2 = r2_score(y_test, y_predict)
print(r2)

print("걸린시간 :", end - start)

# 제출할 놈
y_submit = model.predict(test_csv)
# print(y_submit)                                           # nan이 출력되는 걸로 봐선 test_csv에도 뭔가 결측치가 있었다는 뜻임.
# print(y_submit.shape)  

# .to_csv를 사용해서
# submission_0105.csv를 완성하시오!!!

# print(submission)
submission['count'] = y_submit                              # 대여수를 예측한 카운트에 submission에 넣어준다.
# print(submission)

submission.to_csv(path + 'submission_01111925.csv')         # 날짜.시간 01061152 = 1월 6일 11시 52분

'''
MinMaXScaler

CPU

mse: 2379.456787109375
mae: 36.74049758911133
걸린시간: 11.21312665939331

GPU

mse: 2247.470458984375
mae: 34.58266067504883
걸린시간: 10.701709985733032

...
...

StandardScaler

CPU

mse: 2333.732177734375
mae: 36.00251388549805
걸린시간: 5.643195629119873

GPU

MSE: 2291.377197265625
MAE: 35.412086486816406
걸린시간: 5.598170757293701

'''
