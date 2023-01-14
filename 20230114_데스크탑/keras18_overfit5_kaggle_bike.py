# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np                                                                  # 넘파이를 임포트시키고 간단하게 np로 명시해준다.
import pandas as pd                                                                 # 데이터 분석(Data Analysis)을 위해 널리 사용되는 파이썬 라이브러리 패키지이다.
from tensorflow.keras.models import Sequential                                      # Sequential은 레이어에 순차적으로 연산
from tensorflow.keras.layers import Dense                                           # Dense는 완전 연결층을 구현하는 레이어 모델
from sklearn.model_selection import train_test_split                                # 사이킷런(scikit-learn)의 model_selection 패키지 안에 train_test_split 모듈을 활용하여 손쉽게 train set(학습 데이터 셋)과 test set(테스트 셋)을 분리할 수 있다.
from sklearn.metrics import mean_squared_error, r2_score                            # RMSE 함수는 아직 없어서 직접 만들어 사용. - 회귀 분석 모델 / 사이킷런에서도 rmse는 제공하지 않음. / MSE 함수 불러옴.
                                                                                    # MSE보다 이상치에 덜 민감하다. 이상치에 대한 민감도가 MSE보단 적고 MAE보단 크기 때문에 이상치를 적절히 잘 다룬다고 간주되는 경향이 있다고 한다.

#1. 데이터
path = './_data/bike/'                                                          #./ 현재폴더 /하위폴더 / 하위폴더 /
train_csv = pd.read_csv(path + 'train.csv', index_col=0)                        # 데이터에서 제외할 인덱스 칼럼이 0열에 있음. 열은 데이터가 아닌 인덱스라고 알려줘서 데이터로 들어가는 것을 방지.
                                                                                # train_csv = pd.read_csv('./_data/ddarung/train.csv,' index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)  
submission = pd.read_csv(path + 'samplesubmission.csv', index_col=0)            # ddarung은 submission이었지만, bike는 samplesubmission이기 때문에, 파일명에 알맞게 수정.

print(train_csv)                                                                # [10886 rows x 11 columns]
print(train_csv.shape)                                                          # (10886, 11) / (10886, 11) 이나 타겟인 count가 포함되어 있으므로 피처는 10개이다.
print(submission.shape)                                                         # (6493, 1)         


print(train_csv.columns)                                                        # 컬럼명이 나온다.
                                                                                # (['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                                                                                #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
                                                                                #   dtype='object')

print(train_csv.info())                                                                                                                                  
print(test_csv.info())                                                          # 이 값을 통해 predict 를 할 것이기 때문에 count 값은 필요 없다.
print(train_csv.describe())                                                     # [8 rows x 11 columns]


####결측치 처리 1.삭제####
print(train_csv.isnull().sum())                                                 # data_set의 결측치(Null) 값 총계 출력
                                                                                # 중단점 클릭하고 F5 하면 중단점 직전까지 실행이 된다.

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

train_csv = train_csv.dropna()                                                  # pandas.dropna(): null 값을 포함한 데이터 행 삭제
print(train_csv.isnull().sum())                                                 # 결측값 확인
print(train_csv.shape)                                                          # (10886, 11)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)                   # 작업할 때 axis가 1이면 행, 0이면 열
                                                                                # train.csv와 test.csv를 비교했을 때 casual, registered, count가 있고 없고의 차이임. 이걸 제거해주는 작업임.
                                                                                # column 명이 casual, registered 'count'인 column(axis=1) 삭제
                                                                                # drop function: df.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
                                                                                # labels : 삭제할 레이블명
                                                                                # axis: 0-index처럼 인식, 1-column처럼 인식
                                                                                
                                                                                # drop으로 column을 삭제하는 이유
                                                                                # 'casual', 'registered', 'count'를 예측해도 evaluate할 때 필요가 없으므로 column 삭제

print(x)                                                                        # [10886 rows x 8 columns] -> dropna && drop로 인한 변경
                                                                                
                                                                                
y = train_csv['count']                                                          # train_csv에서 count col 추출 - pandas의 기능

print(y)                                                                        # (10886, 11)
print(y.shape)                                                                  # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=1234
)

print(x_train.shape, x_test.shape)                                              # (7620, 8) (3266, 8)
print(y_train.shape, y_test.shape)                                              # (7620,) (3266,)

# 2. 모델구성
model = Sequential()                                                            # 모델은 순차적으로 구성하겠다.
model.add(Dense(32, input_dim=8, activation='linear'))                          # 8열이라서 iuput_dim은 8       # x열의 개수
model.add(Dense(50))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='relu'))                       
model.add(Dense(1))                                                             # output_dim = 1
                                                                                # Activation default: linear
                                                                                # output_dim에서 activation을 'sigmoid'를 사용할 경우, return value가 0(predict값이 0.5미만일 경우) or 1(predict값이 0.5이상일 경우)로만 반환하게 됨.
                                                                                # → 이진 분류에서만 sigmoid를 마지막에 사용. / 마지막에 sigmoid를 쓰면 안 된다. 0과 1로 다 바뀌기 때문이다. 단, 2진 분류에서만 쓴다.
                                                                                # Hidden Layer에서 sigmoid를 사용할 수 있으나, 값이 너무 한정적으로 변하기때문에 이진 분류를 제외한 곳에서 사용을 권장하지 않음.
                                                                                # relu도 hidden layer에서만 사용 권장, output_dim에서 사용 시, 음수값이 왜곡될 가능성이 있음. / relu는 통상적으로 중간 Hidden layer에서 사용한다. 마지막에서 사용하면 안 된다.
                                                                                

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
plt.title('bike loss')                                            # 제목을 지정
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

submission.to_csv(path + 'submission_01141413.csv')