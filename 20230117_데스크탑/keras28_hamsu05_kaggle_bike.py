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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size=0.7, shuffle=True, random_state=1234
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)                                                             # x_train에 대한 범위의 가중치 생성
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
x_train = scaler.fit_transform(x_train)                                         # 한 줄로 정리
x_test = scaler.transform(x_test)                               
test_csv = scaler.transform(test_csv)


print(x_train.shape, x_test.shape)                                              # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape)                                              # (1021, ) (438, )

#2. 모델구성
model=Sequential()                                                                  # 모델은 순차적으로 구성하겠다.
model.add(Dense(48, activation="relu", input_dim=8))                               # 13열이라서 iuput_dim은 13       # x열의 개수      # relu: 0 이하의 값은 다음 레이어에 전달하지 않습니다. 0 이상의 값은 그대로 출력합니다.
model.add(Dense(32, activation="relu"))                               
model.add(Dense(16, activation="relu"))                                             # Dense값을 임의로 조절
model.add(Dense(8, activation="relu"))                                                       
model.add(Dense(1))                                                                 # output=1                       # y열의 개수

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                        # MSE는 추정된 값과 실제 값 간의 평균 제곱 차이를 의미한다. #주로 회귀에서 사용되는 손실 함수이며, 정확도 개념은 회귀에 적용되지 않는다고 한다.
                                                                                                                # 일반적인 회귀 지표는 MAE이며, MSE는 손실 함수로 쓰이고 MAE는 회귀지표로써 사용된다.
                                                                                                                # mae = mean(평균),absolute(절대값),error / mse = mean loss mse(평균제곱오차)
                                                                                                                # loss : 훈련에 영향을 미친다. loss는 다음 가중치에 반영 → 반복 훈련
                                                                                    
                                                                                    # metrics = 어떤 방식으로 모델을 돌릴 것인가? 즉, loss = 손실함수, metrics = 평가지표
                                                                                    # metrics에 사용하는 ['mae', 'mse', 'accuracy', 'acc']는 훈련에 영향을 미치지 않는다. 참고용으로 사용하는데, 사용방법은 mae 또는 mse 아니면 ['mae', 'mse'] 이렇게 여러개의 리스트를 써도 된다.
                                                                                    # 'accuracy' = 'acc'
                                                            
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.25)        # epoch: 훈련을 여러번 시키는것. fit: 훈련시킴. / epoch 너무 많이하면 오히려 성능이 떨어지는 경우가 있기 때문에, 적절한 epoch 수치를 찾아내야 함. / 이번엔 훈련 수치를 200번으로 조정.
                                                                                    # batch_size가 크면 속도가 빨라지고 정확도가 떨어짐. 과적합(overfitting)을 고려해서 조절해야 함. # batch size default=32(기본값) # 여기서는 기본값으로 임의로 세팅
                                                                                    # x에 대한 예상문제를 평가하는 과정을 추가 (validation_data)
                                                                                    # 데이터 검증 (훈련하고 검증하고)
                                                                                    # 훈련 + '검증(Validation)' + 평가 (fit + 'validation'+ evaluate)
                                                                                    # validation_data를 통해서 val_loss 추가 / val_loss 즉, 검증할 때 손실값이 출력된다. / 기준을 잡을 때, val_loss로 기준을 잡는다.
                                                                                    # 훈련(train)보다 검증(validation)결과를 기준으로 테스트 결과를 판단해야 함.
                                                                                    # validation_split을 통해서 x_train과 y_train 중 0.25의 validation 값 지정.
                                                                                    

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                               # loss 값으로 평가, 반환 # evaluate : 들어가는 데이터는 훈련데이터가 들어가면 안된다. # loss 수치가 낮으면 가중치에 최적화. 판단의 기준은 predict보다 loss(0과 가까울수록 좋다)이다.
                                                                                    # evaluate 에도 배치 사이즈가 존재. defalut 값 = 32
                                                                                    # test loss가 train loss보다 안좋다.
print('loss : ', loss)

y_predict = model.predict(x_test)                                                   # x의 전체 값을 예측해서 y_predict

print("============================")
print(y_test)
print(y_predict)
print("============================")

def rmse(y_test, y_predict):                                                        # rmse를 직접 만들기 위해 함수를 선언하는 과정이며 y_test, y_predict로 만든 RMSE 함수
    return np.sqrt(mean_squared_error(y_test, y_predict))                           # mse에 root 씌워서 rmse 만든 것. / y_test, y_predict로 만든 MSE에 루트(sqrt)를 씌워서 내놔라.

print('RMSE :', rmse(y_test, y_predict))                                            # 값이 낮을수록 정밀도가 높음 

r2 = r2_score(y_test, y_predict)                                                    # R2 = 정확도와 비슷한 개념 / mse와 반대로 값이 높을수록 좋은 성능의 모델이다.
                                                                                    # max값 1에 가까울 수록 설명력(정확도)이 높음.

print('R2 : ', r2)

# 제출할 놈
y_submit = model.predict(test_csv)
                                                                                # print(y_submit)       # nan이 출력되는 걸로 봐선 test_csv에도 뭔가 결측치가 있었다는 뜻임.
                                                                                # print(y_submit.shape)   # (715, 1)

# .to_csv를 사용해서
# submission_0105.csv를 완성하시오!!!

# print(submission)
submission['count'] = y_submit                                                  # 대여수를 예측한 카운트에 submission에 넣어준다.
# print(submission)3


submission.to_csv(path + 'submission_01171905.csv')


'''

CPU 결과(minmax)
RMSE : 148.0289925257338
R2 :  0.3132065269801573

GPU 결과(minmax)
RMSE : 147.98523370014533
R2 :  0.3136125127707322

CPU 결과(stand)
RMSE : 146.38314397696448
R2 :  0.32839374299542856

GPU 결과(stand)
RMSE : 146.23401167289597
R2 :  0.3297614847477309
'''