# https://www.kaggle.com/competitions/bike-sharing-demand

# 만들어봐!

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # 데이터에서 제외할 인덱스 칼럼이 0열에 있음. 열은 데이터가 아닌 인덱스라고 알려줘서 데이터로 들어가는 것을 방지.
# train_csv = pd.read_csv('./_data/ddarung/train.csv,' index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)       # index_col 지정하지 않으면, 인덱스가 자동으로 생성된다.
submission = pd.read_csv(path + 'samplesubmission.csv', index_col=0)

print(train_csv)   # [10886 rows x 11 columns]
print(train_csv.shape)  # (10886, 11)
print(submission.shape) # (6493, 1)

print(train_csv.columns)
# (['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object')
print(train_csv.info()) # 
print(test_csv.info()) # 
print(train_csv.describe())     # [8 rows x 11 columns]

#### 결측치 처리 1. 제거 ####
print(train_csv.isnull().sum()) # 중단점 클릭하고 F5 하면 중단점 직전까지 실행이 된다.

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
print(train_csv.isnull().sum())     # 결측값 확인
print(train_csv.shape)  # (10886, 11)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)       # 작업할 때 axis가 1이면 행, 0이면 열
# train.csv와 test.csv를 비교했을 때 casual, registered, count가 있고 없고의 차이임. 이걸 제거해주는 작업임.

print(x)   # [10886 rows x 10 columns]
y = train_csv['count']
print(y)   # (10886, 11)
print(y.shape)  # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=1234
)

print(x_train.shape, x_test.shape)  # (7620, 10) (3266, 10)
print(y_train.shape, y_test.shape)  # (7620,) (3266,)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))  # default값은 liner
model.add(Dense(64, activation='relu'))
model.add(Dense(144, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))   # Dense 값을 임의로 조절해 주었다.
model.add(Dense(1, activation='linear')) # output 
# 마지막에 sigmoid를 쓰면 안 된다. 0과 1로 다 바뀌기 때문이다. 단, 2진 분류에서만 쓴다.
# relu는 통상적으로 중간 Hidden layer에서 사용한다. 마지막에서 사용하면 안 된다.

#3 컴파일, 훈련
import time     # 시간을 임포트
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])  # 가급적 유사지표에서는 mse
start = time.time()     # 시작 시간
model.fit(x_train, y_train, epochs=200, batch_size=32)
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

submission.to_csv(path + 'submission_01061222.csv')     # 날짜.시간 01061152 = 1월 6일 11시 52분

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
model.add(Dense(32, input_dim=8, activation='relu'))  # default값은 liner
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))   # Dense 값을 임의로 조절해 주었다.
model.add(Dense(1, activation='linear')) # output 
# 마지막에 sigmoid를 쓰면 안 된다. 0과 1로 다 바뀌기 때문이다. 단, 2진 분류에서만 쓴다.
# relu는 통상적으로 중간 Hidden layer에서 사용한다. 마지막에서 사용하면 안 된다.

'''

'''
model.fit(x_train, y_train, epochs=10, batch_size=32) 

# cpu 걸린시간 : 3.1350724697113037   // relu
# RMSE:  153.54143216585254

# gpu 걸린시간 : 3.1396443843841553    // relu
# RMSE:  156.9311806496223

'''

'''
model.fit(x_train, y_train, epochs=100, batch_size=32)

# cpu 걸린시간 : 30.024340629577637   // relu
# RMSE:  148.3596695039428

# gpu 걸린시간 : 28.324995279312134  // relu
# RMSE:  149.14917531853322
'''

'''
model.fit(x_train, y_train, epochs=300, batch_size=32)

# cpu 걸린시간 : 79.33622574806213   // relu
# RMSE:  152.54737979668278

# gpu 걸린시간 : 80.34421920776367  // relu
# RMSE:  147.9439870274285
'''

'''
#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))  # default값은 liner
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))   # Dense 값을 임의로 조절해 주었다.
model.add(Dense(1, activation='linear')) # output 
# 마지막에 sigmoid를 쓰면 안 된다. 0과 1로 다 바뀌기 때문이다. 단, 2진 분류에서만 쓴다.
# relu는 통상적으로 중간 Hidden layer에서 사용한다. 마지막에서 사용하면 안 된다.

model.fit(x_train, y_train, epochs=300, batch_size=32)

# cpu 걸린시간 : 99.00116610527039   // relu
# RMSE:  149.90845962822567

# gpu 걸린시간 : 102.45602989196777 // relu
# RMSE:  150.4282481628387
'''

'''
model.fit(x_train, y_train, epochs=10, batch_size=32)

# cpu 걸린시간 : 3.901611566543579   // relu
# RMSE:  152.86236728427752

# gpu 걸린시간 : 4.081185579299927 // relu
# RMSE:  152.90431795734142
'''

'''
model.fit(x_train, y_train, epochs=500, batch_size=32)

# cpu 걸린시간 : 3.901611566543579   // relu
# RMSE:  152.86236728427752

# gpu 걸린시간 : 4.081185579299927 // relu
# RMSE:  152.90431795734142
'''

'''
model.fit(x_train, y_train, epochs=300, batch_size=32)

# cpu 걸린시간 : 101.21683597564697   // relu
# RMSE:  153.85554296083365

# gpu 걸린시간 : 102.60233736038208 // relu
# RMSE:  153.50022353322237
'''

'''
model.fit(x_train, y_train, epochs=200, batch_size=32)

# cpu 걸린시간 : 101.21683597564697   // relu
# RMSE:  153.85554296083365

# gpu 걸린시간 : 102.60233736038208 // relu
# RMSE:  153.50022353322237
'''