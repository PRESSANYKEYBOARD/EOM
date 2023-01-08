# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np                                                      
import pandas as pd                                                             # 데이터 분석(Data Analysis)을 위해 널리 사용되는 파이썬 라이브러리 패키지이다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)                        # 데이터에서 제외할 인덱스 칼럼이 0열에 있음. 열은 데이터가 아닌 인덱스라고 알려줘서 데이터로 들어가는 것을 방지.
                                                                                # train_csv = pd.read_csv('./_data/ddarung/train.csv,' index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)  
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)                                                          # (1459, 10) 카운트 분리하면 (1459, 9)
print(submission.shape)                                                         # (715, 1)  715개라는 평가 데이터를 알아야 되기 때문에, 삭제하면 안 된다.


print(train_csv.columns)                                                        # 컬럼명이 나온다.
                                                                                # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                                                                                #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                                                                                #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                                                                                #       dtype='object')

print(train_csv.info())                                                         # 결측치 데이터가 2개가 있음. 임의의 데이터를 넣으면 오차가 커지기 때문에 제거해야 한다. 단, 데이터가 적을시에는 삭제하는 것이 오히려 치명적이다. / 결측치 : 1459 - 1457 = 2개 데이터 빠짐
                                                                                #  0   hour                    1459 non-null   int64  
                                                                                #  1   hour_bef_temperature    1457 non-null   float64
                                                                                #  2   hour_bef_precipitation  1457 non-null   float64
                                                                                
print(test_csv.info())                                                          # 이 값을 통해 predict 를 할 것이기 때문에 count 값은 필요 없다.

print(train_csv.describe())                                                     #     hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  ...  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5        count
                                                                                # count  1459.000000           1457.000000             1457.000000         1450.000000  ...     1383.000000    1369.000000     1342.000000  1459.000000
                                                                                # mean     11.493489             16.717433                0.031572            2.479034  ...        0.039149      57.168736       30.327124   108.563400
                                                                                # std       6.922790              5.239150                0.174917            1.378265  ...        0.019509      31.771019       14.713252    82.631733
                                                                                # min       0.000000              3.100000                0.000000            0.000000  ...        0.003000       9.000000        8.000000     1.000000
                                                                                # 25%       5.500000             12.800000                0.000000            1.400000  ...        0.025500      36.000000       20.000000    37.000000
                                                                                # 50%      11.000000             16.600000                0.000000            2.300000  ...        0.039000      51.000000       26.000000    96.000000
                                                                                # 75%      17.500000             20.100000                0.000000            3.400000  ...        0.052000      69.000000       37.000000   150.000000


####결측치 처리 1.삭제####
print(train_csv.isnull().sum())                                                 # train_csv에 넓값을 출력
                                                                                # 중단점 클릭하고 F5 하면 중단점 직전까지 실행이 된다.

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

print(train_csv.shape) 
train_csv = train_csv.dropna()                                                  # train_csv에 있는 널값을 삭제하는 메서드 = .dropna()
x = train_csv.drop(['count'],axis=1)                                            # train_csv 에서 count컬럼제거.  drop메소드는 컬럼제거.
print(x)                                                                        # [1459 rows x 9 columns]  (1459,9)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=1234
)

print(x_train.shape, x_test.shape)                                              # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape)                                              # (1021, ) (438, )

#2. 모델구성
model = Sequential()                                                            # 모델을 순차적으로 구성하겠다.
model.add(Dense(32, input_dim=9))                                               # 9열이라서 iuput_dim은 9       # x열의 개수  
model.add(Dense(50))
model.add(Dense(128))
model.add(Dense(256, activation='relu'))                                        # relu: 0 이하의 값은 다음 레이어에 전달하지 않습니다. 0 이상의 값은 그대로 출력합니다.
model.add(Dense(128, activation='relu'))
model.add(Dense(16))
model.add(Dense(1))                                                             # output=1                       # y열의 개수

#3 컴파일, 훈련
import time                                                                     # 시간을 임포트 / time() 메서드를 사용하기위한 class import
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])                                                  # 가급적 유사지표에서는 mse
start = time.time()                                                             # 시작 시간
model.fit(x_train, y_train, epochs= 00, batch_size=1, validation_split=0.15)                            # epoch: 훈련을 여러번 시키는것. fit: 훈련시킴. / epoch 너무 많이하면 오히려 성능이 떨어지는 경우가 있기 때문에, 적절한 epoch 수치를 찾아내야 함. / 이번엔 훈련 수치를 100번으로 조정.
                                                                                                        # batch_size가 크면 속도가 빨라지고 정확도가 떨어짐. 과적합(overfitting)을 고려해서 조절해야 함. # batch size default=32(기본값) # 여기서는 1로 잡아줌.
end = time.time()                                                                                       # 종료 시간

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                           # x_test, y_test 값으로 평가
print('loss:', loss)

y_predict = model.predict(x_test)                                               # RMSE 출력을 위해 model.predict를 명시해준다. / x_test값으로 y_predict 예측
print(y_predict)                                                                # 실행하면 loss 값이 nan이라고 뜰 것인데, 뭔가 연산이 안 되고 있다는 뜻임. 없는 데이터(결측치)가 있었기 때문.

                                                                                # 결측치 때문에 loss 값에 nan이라고 출력되고 실패.

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
submission['count'] = y_submit                                                  # 대여수를 예측한 카운트에 submission에 넣어준다.
# print(submission)

submission.to_csv(path + 'submission_01081339.csv')

'''
내가 찾은 답
result = pd.read_csv(path + 'submission.csv', index_col=0)
result['count'] = y_submit
result =.to_csv(path + 'submission_0105.csv', index = true)

'''

'''
CPU

걸린시간: 72.88101959228516
RMSE: 57.33609136526057

'''
   
'''
GPU

걸린시간: 70.03609776496887
RMSE: 54.80649485559607

'''
