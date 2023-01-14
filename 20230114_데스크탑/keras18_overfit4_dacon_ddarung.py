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

#3. 컴파일, 훈련
import time
start = time.time()
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=50, batch_size=1,
          validation_split=0.2,
          verbose=1)                                                # verbose= 진행표시줄 on/off
                                                                    # default=1
                                                                    # hist = history
                                                                    # model.fit 값을 받아서 hist라는 변수로 리턴한다.

                                                                    # True : 1, False : 0, 프로그래스바 제거(진행바 사라짐) : 2, 에포(반복치)만 보여줌 : 3 ~
                                                                    # 말수가 많음 실행할 때, 코드 보여주는 게 딜레이가 생긴다.
                                                                    # 자원낭비
                                                                    
                                                                    # 함수 수행 시 발생하는 상세한 정보들을 표준 출력으로 자세히 내보낼 것인가를 나타냄
                                                                    # 0: 미출력, 1(Default): 자세히, 2: 함축적 정보 출력 3. 2보다 더 함축적 정보 출력
                                                                    
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

print("걸린시간:", end - start)                                      # verbose = 학습 중 출력되는 문구를 설정합니다.
                                                                        # - 0 : 아무 것도 출력하지 않습니다.
                                                                        # - 1 : 훈련의 진행도를 보여주는 진행 막대를 보여줍니다.
                                                                        # - 2 : 미니 배치마다 손실 정보를 출력합니다.

                                                                    # verbose=1 / loss: 47.06433868408203, 걸린시간: 13.293114423751831
                                                                    # verbose=0 / loss: 53.02677536010742, 걸린시간: 10.317590236663818
                                                                    # verbose=2 / loss: 57.73585891723633, 걸린시간: 10.568190813064575
                                                                    # verbose=3 / loss: 58.02903366088867, 걸린시간: 10.319077253341675

print("==============================")                                               
print(hist)                                                         # <keras.callbacks.History object at 0x000001BF527C0AF0>
print("==============================")
print(hist.history)                                                 # hist 공간 안에 history 라는 제공된 변수가 있다.
                                                                    # loss와 val_loss 값이 딕셔너리 형태로 들어가 있다.
                                                                    # 파이썬의 데이터 형태는 list / key / value
                                                                    # dictionary(딕셔너리): 형태와 {} 형태로 묶여져 있는데, value 값이 list 형태로 묶여져 있다.
                                                                    # 딕셔너리는 key와 value 형태 / 홍길동: {국영수 점수} / 심청이 :{국영수 점수}
                                                                    # 두 개 이상=list 형태
                                                                    
                                                                    # model.fit은 훈련의 결과값을 반환하고 그걸 hist(history)라 하자.
                                                                    # print(hist.history)하면 loss, val_loss, metrics 등을 dictionary 형태로 보여준다.
                                                                    # dictionary : {'분류이름(key)' : [ , , , , ...], 'val_loss' : [ , , , , ...] (value)...} : key, value 형태이다.
                                                                    
print("==============================")
print(hist.history['loss'])                                         # loss 값 출력
print("==============================")
print(hist.history['val_loss'])                                     # val_loss 값 출력

import matplotlib.pyplot as plt                                     # (epochs, loss)의 산점도 및 그래프를 작성할 수 있음

plt.figure(figsize=(9,6))                                           # 그림에 대한 판 사이즈
plt.plot(hist.history['loss'], c='red', 
         marker='.', label='loss')                                  # list 형태에서 x 명시는 굳이 안 해도 상관 없다. y만 넣어주면 된다.
                                                                    # c= 색깔 지정
                                                                    # marker= 선의 모양
                                                                    # label= 선의 이름
                                                
plt.plot(hist.history['val_loss'], c='blue', 
         marker='.', label='val_loss')              
               
plt.grid()                                                          # 격자 넣기
plt.xlabel('epochs')                                                # x축을 epochs 지정
plt.ylabel('loss')                                                  # y축을 loss 지정
plt.title('ddarung loss')                                            # 제목을 지정
plt.legend()                                                        # 라벨이 나오게 된다 / 그래프가 없는 지점에 나온다.
# plt.legend(loc='upper left')                                      # loc= 라벨을 어디에 나오게? / upper right
plt.show()                                                          # 그림 보여주기

# 제출할 놈
y_submit = model.predict(test_csv)
                                                                                # print(y_submit)       # nan이 출력되는 걸로 봐선 test_csv에도 뭔가 결측치가 있었다는 뜻임.
                                                                                # print(y_submit.shape)   # (715, 1)

# .to_csv를 사용해서
# submission_0105.csv를 완성하시오!!!

# print(submission)
submission['count'] = y_submit                                                  # 대여수를 예측한 카운트에 submission에 넣어준다.
# print(submission)

submission.to_csv(path + 'submission_01141400.csv')

