import numpy as np
from tensorflow.keras.models import Sequential, Model                           # model은 인풋 레이어를 명시해줘야함.
from tensorflow.keras.layers import Dense, Input
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

#1. 데이터
# path = './_data/bike/'
# path = '../_data/bike/'
path = 'C:/study/_data/bike/'                                   # 절대 경로
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # 데이터에서 제외할 인덱스 칼럼이 0열에 있음. 열은 데이터가 아닌 인덱스라고 알려줘서 데이터로 들어가는 것을 방지.
                                                                # train_csv = pd.read_csv('./_data/ddarung/train.csv,' index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)           # index_col 지정하지 않으면, 인덱스가 자동으로 생성된다.
submission = pd.read_csv(path + 'samplesubmission.csv', index_col=0)

print(train_csv)                                                # [10886 rows x 11 columns]
print(train_csv.shape)                                          # (10886, 11)
print(submission.shape)                                         # (6493, 1)

print(train_csv.columns)
                                                                # (['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                                                                    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
                                                                    #   dtype='object')
print(train_csv.info()) 
print(test_csv.info()) 
print(train_csv.describe())                                     # [8 rows x 11 columns]

#### 결측치 처리 1. 제거 ####
print(train_csv.isnull().sum())                                 # 중단점 클릭하고 F5 하면 중단점 직전까지 실행이 된다.

'''
결측치 데이터
season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
casual        0
registered    0
count         0

'''

train_csv = train_csv.dropna()
print(train_csv.isnull().sum())                                     # 결측값 확인
print(train_csv.shape)  # (10886, 11)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)       # 작업할 때 axis가 1이면 행, 0이면 열
                                                                    # train.csv와 test.csv를 비교했을 때 casual, registered, count가 있고 없고의 차이임. 이걸 제거해주는 작업임.

print(x)                                                            # [10886 rows x 10 columns]
y = train_csv['count']
print(y)                                                            # (10886, 11)
print(y.shape)                                                      # (10886,)
                                                           
x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, test_size=0.2 
)                                                               # train은 대략 8,708개 정도

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)                                             # x_train에 대한 범위의 가중치 생성
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
x_train = scaler.fit_transform(x_train)                         # 한 줄로 정리
x_test = scaler.transform(x_test)                               
test_csv = scaler.transform(test_csv)


# #2. 모델구성
# model = Sequential()
# model.add(Dense(5, input_shape=(8, )))                          # 스칼라가 8개로 볼 수 있기 때문에, (8, ) 라고 해석할 수 있다.
#                                                                 # (100, 10, 5) 라는 데이터가 들어가면, 행 무시 열 우선이기 때문에 (10, 5)로 들어감.
#                                                                 # 앞으로는 이렇게 쓰자!!!
# model.add(Dense(40, activation='sigmoid'))
# model.add(Dense(120, activation='relu'))
# model.add(Dense(30, activation='linear'))
# model.add(Dense(1))

# 2. 모델구성(함수형)
input1 = Input(shape=(8, ))                                     
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
model.fit(x_train, y_train, epochs=100, batch_size=32,
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
# print(y_submit)      
# print(y_submit.shape)   

# .to_csv를 사용해서
# submission_0105.csv를 완성하시오!!!

# print(submission)
submission['count'] = y_submit                              
# print(submission)

submission.to_csv(path + 'submission_01111937.csv')         # 날짜.시간 01061152 = 1월 6일 11시 52분

'''
MinMaXScaler

CPU

mse: 20808.875
mae: 108.30915832519531
걸린시간: 28.73825716972351

GPU

mse: 20676.115234375
mae: 107.95184326171875
걸린시간: 26.933178901672363

...
...

StandardScaler

CPU

mse: 21300.318359375
mae: 108.35272979736328
걸린시간: 27.99848437309265

GPU

mse: 21291.109375
mae: 108.43633270263672
걸린시간: 30.706363201141357

'''