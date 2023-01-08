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
                                                                                

#3 컴파일, 훈련
import time                                                                     # 시간을 임포트 / time() 메서드를 사용하기위한 class import
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])                                                  # 가급적 유사지표에서는 mse
start = time.time()                                                             # 시작 시간
model.fit(x_train, y_train, epochs=500, batch_size=16, validation_split=0.25)   # epoch: 훈련을 여러번 시키는것. fit: 훈련시킴. / epoch 너무 많이하면 오히려 성능이 떨어지는 경우가 있기 때문에, 적절한 epoch 수치를 찾아내야 함. / 이번엔 훈련 수치를 500번으로 조정.
                                                                                # batch_size가 크면 속도가 빨라지고 정확도가 떨어짐. 과적합(overfitting)을 고려해서 조절해야 함. # batch size default=32(기본값) # 여기서는 16으로 잡아줌.
end = time.time()                                                               # 종료 시간

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                           # x_test, y_test 값으로 평가
                                                                                # train data 중 test set를 evaluate → loss 판단
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
                                                                                # print(y_submit)           # nan이 출력되는 걸로 봐선 test_csv에도 뭔가 결측치가 있었다는 뜻임.
                                                                                # print(y_submit.shape)     # (6493, 1)

# .to_csv를 사용해서
# submission_0105.csv를 완성하시오!!!

# print(submission)
submission['count'] = y_submit                                                  # 대여수를 예측한 카운트에 submission에 넣어준다.
# print(submission)

submission.to_csv(path + 'submission_01081406.csv')                             # 날짜.시간 01061152 = 1월 6일 11시 52분

'''
내가 찾은 답
result = pd.read_csv(path + 'submission.csv', index_col=0)
result['count'] = y_submit
result =.to_csv(path + 'submission_0105.csv', index = true)

'''

'''
x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=1234
)

# 2. 모델구성
model = Sequential()                                                            # 모델은 순차적으로 구성하겠다.
model.add(Dense(32, input_dim=8, activation='linear'))                          # 8열이라서 iuput_dim은 8       # x열의 개수
model.add(Dense(50))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))                                                             # output_dim = 1
                                                                                # Activation default: linear
                                                                                # output_dim에서 activation을 'sigmoid'를 사용할 경우, return value가 0(predict값이 0.5미만일 경우) or 1(predict값이 0.5이상일 경우)로만 반환하게 됨.
                                                                                # → 이진 분류에서만 sigmoid를 마지막에 사용.
                                                                                # Hidden Layer에서 sigmoid를 사용할 수 있으나, 값이 너무 한정적으로 변하기때문에 이진 분류를 제외한 곳에서 사용을 권장하지 않음.
                                                                                # relu도 hidden layer에서만 사용 권장, output_dim에서 사용 시, 음수값이 왜곡될 가능성이 있음.

model.fit(x_train, y_train, epochs=500, batch_size=16, validation_split=0.25)   # epoch: 훈련을 여러번 시키는것. fit: 훈련시킴. / epoch 너무 많이하면 오히려 성능이 떨어지는 경우가 있기 때문에, 적절한 epoch 수치를 찾아내야 함. / 이번엔 훈련 수치를 500번으로 조정.
                                                                                # batch_size가 크면 속도가 빨라지고 정확도가 떨어짐. 과적합(overfitting)을 고려해서 조절해야 함. # batch size default=32(기본값) # 여기서는 16으로 잡아줌.

'''

'''
CPU

걸린시간: 162.6117377281189
RMSE: 151.46713146897727

'''
   
'''
GPU

걸린시간: 166.14476490020752
RMSE: 150.97476314213594

'''
