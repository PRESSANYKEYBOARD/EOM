# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np                                                                  # 넘파이를 임포트시키고 간단하게 np로 명시해준다.
import tensorflow as tf                                                             # 텐서플로를 임포트시키고 tf라고 명시
from tensorflow.keras.models import Sequential, Model                               # Sequential은 레이어에 순차적으로 연산
from tensorflow.keras.layers import Dense, Input                                    # Dense는 완전 연결층을 구현하는 레이어 모델
from sklearn.model_selection import train_test_split                                # 사이킷런(scikit-learn)의 model_selection 패키지 안에 train_test_split 모듈을 활용하여 손쉽게 train set(학습 데이터 셋)과 test set(테스트 셋)을 분리할 수 있다.
from sklearn.metrics import mean_squared_error, r2_score                            # RMSE 함수는 아직 없어서 직접 만들어 사용. - 회귀 분석 모델 / 사이킷런에서도 rmse는 제공하지 않음. / MSE 함수 불러옴.
                                                                                    # MSE보다 이상치에 덜 민감하다. 이상치에 대한 민감도가 MSE보단 적고 MAE보단 크기 때문에 이상치를 적절히 잘 다룬다고 간주되는 경향이 있다고 한다.
from sklearn.datasets import load_boston                                            # sklearn에서 제공하는 load_boston 가져오기
from sklearn.preprocessing import MinMaxScaler, StandardScaler

dataset= load_boston()
x = dataset.data                                                                    # 집에 대한 데이터
y = dataset.target                                                                  # 집 값

# scaler = MinMaxScaler()                                                           # 정규화
scaler = StandardScaler()                                                           # 표준화
scaler.fit(x)                                                                       # x의 값의 범위만큼 가중치 생성
x = scaler.transform(x)                                                             # 생성된 가중치만큼의 data 형태 변형
print(x)
print(type(x))                                                                      # <class 'numpy.ndarray'>

'''
print(x)
print(x.shape) #(506,13)

print(y)
print(y.shape) #(506, )

print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

print(dataset.DESCR)                                                                # 데이터셋의 description 볼 수 있음.

'''

x_train, x_test, y_train, y_test = train_test_split(                                # 파라미터 (x와 y에 값을 대입) 
    x, y, train_size=0.7, random_state=123)                                         # 데이터셋에서 70%(x_train) # 데이터셋에서 30%(x_test) / 둘 중 하나만 명시해주면 된다. 
                                                                                    # test_size default=0.25
                                                                                    # shuffle=True 무작위 추출, False=순차적 추출 / default=True
                                                                                    # random_state로 잡아주면 그다음 데이터도 동일한 데이터로 들어감. 아무런 의미 없는 값을 넣어도 상관없다.

##2. 모델구성(순차형)
# model = Sequential()
# model.add(Dense(50, activation='relu', input_shape=(13,)))
# model.add(Dense(40, activation='sigmoid'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='linear'))
# model.add(Dense(1, activation='linear'))
# model.summary()
# # Total params: 4,611                                                             # 총 연산량: 4611

# 2. 모델구성(함수형)
input1 = Input(shape=(13, ))                                                        # 인풋레이어를 13 레이어에 준다
Dense1 = Dense(50, activation='relu')(input1)
Dense2 = Dense(64, activation='sigmoid')(Dense1)
Dense3 = Dense(32, activation='relu')(Dense2)
Dense4 = Dense(16, activation='linear')(Dense3)
output1 = Dense(1, activation='linear')(Dense4)
model = Model(inputs=input1, outputs=output1)                                   # 시작하는 부분과 끝 부분이 어디인지 알려주는 부분
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['mae'])                        # MSE는 추정된 값과 실제 값 간의 평균 제곱 차이를 의미한다. #주로 회귀에서 사용되는 손실 함수이며, 정확도 개념은 회귀에 적용되지 않는다고 한다.
                                                                                                                # 일반적인 회귀 지표는 MAE이며, MSE는 손실 함수로 쓰이고 MAE는 회귀지표로써 사용된다.
                                                                                                                # mae = mean(평균),absolute(절대값),error / mse = mean loss mse(평균제곱오차)
                                                                                                                # loss : 훈련에 영향을 미친다. loss는 다음 가중치에 반영 → 반복 훈련
                                                                                    
                                                                                    # metrics = 어떤 방식으로 모델을 돌릴 것인가? 즉, loss = 손실함수, metrics = 평가지표
                                                                                    # metrics에 사용하는 ['mae', 'mse', 'accuracy', 'acc']는 훈련에 영향을 미치지 않는다. 참고용으로 사용하는데, 사용방법은 mae 또는 mse 아니면 ['mae', 'mse'] 이렇게 여러개의 리스트를 써도 된다.
                                                                                    # 'accuracy' = 'acc'
                                                            
model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.25)        # epoch: 훈련을 여러번 시키는것. fit: 훈련시킴. / epoch 너무 많이하면 오히려 성능이 떨어지는 경우가 있기 때문에, 적절한 epoch 수치를 찾아내야 함. / 이번엔 훈련 수치를 200번으로 조정.
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


'''
x_train, x_test, y_train, y_test = train_test_split(                                # 파라미터 (x와 y에 값을 대입) 
    x, y, train_size=0.7, random_state=123)                                         # 데이터셋에서 70%(x_train) # 데이터셋에서 30%(x_test) / 둘 중 하나만 명시해주면 된다. 
                                                                                    # test_size default=0.25
                                                                                    # shuffle=True 무작위 추출, False=순차적 추출 / default=True
                                                                                    # random_state로 잡아주면 그다음 데이터도 동일한 데이터로 들어감. 아무런 의미 없는 값을 넣어도 상관없다.
#2. 모델구성
model=Sequential()                                                                  # 모델은 순차적으로 구성하겠다.
model.add(Dense(48, activation="relu", input_dim=13))                               # 13열이라서 iuput_dim은 13       # x열의 개수      # relu: 0 이하의 값은 다음 레이어에 전달하지 않습니다. 0 이상의 값은 그대로 출력합니다.
model.add(Dense(32, activation="relu"))                               
model.add(Dense(16, activation="relu"))                                             # Dense값을 임의로 조절
model.add(Dense(8, activation="relu"))                                                       
model.add(Dense(1))                                                                 # output=1                       # y열의 개수

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['mae'])                        # MSE는 추정된 값과 실제 값 간의 평균 제곱 차이를 의미한다. #주로 회귀에서 사용되는 손실 함수이며, 정확도 개념은 회귀에 적용되지 않는다고 한다.
                                                                                                                # 일반적인 회귀 지표는 MAE이며, MSE는 손실 함수로 쓰이고 MAE는 회귀지표로써 사용된다.
                                                                                                                # mae = mean(평균),absolute(절대값),error / mse = mean loss mse(평균제곱오차)
                                                                                                                # loss : 훈련에 영향을 미친다. loss는 다음 가중치에 반영 → 반복 훈련
                                                                                                                
model.fit(x_train, y_train, epochs=200, batch_size=1, validation_split=0.25)        # epoch: 훈련을 여러번 시키는것. fit: 훈련시킴. / epoch 너무 많이하면 오히려 성능이 떨어지는 경우가 있기 때문에, 적절한 epoch 수치를 찾아내야 함. / 이번엔 훈련 수치를 200번으로 조정.
                                                                                    # batch_size가 크면 속도가 빨라지고 정확도가 떨어짐. 과적합(overfitting)을 고려해서 조절해야 함. # batch size default=32(기본값) # 여기서는 기본값으로 임의로 세팅
                                                                                    # x에 대한 예상문제를 평가하는 과정을 추가 (validation_data)
                                                                                    # 데이터 검증 (훈련하고 검증하고)
                                                                                    # 훈련 + '검증(Validation)' + 평가 (fit + 'validation'+ evaluate)
                                                                                    # validation_data를 통해서 val_loss 추가 / val_loss 즉, 검증할 때 손실값이 출력된다. / 기준을 잡을 때, val_loss로 기준을 잡는다.
                                                                                    # 훈련(train)보다 검증(validation)결과를 기준으로 테스트 결과를 판단해야 함.
                                                                                    # validation_split을 통해서 x_train과 y_train 중 0.25의 validation 값 지정.


CPU 결과(minmax)
RMSE : 4.176586346932155
R2 :  0.7841855083740839

GPU 결과(minmax)
RMSE : 4.28723154243028
R2 :  0.7725994277720056

CPU 결과(stand)
RMSE : 4.1130384039159384
R2 :  0.7907029033835759

GPU 결과(stand)
RMSE : 4.240886803297804
R2 :  0.7774892304612162

'''