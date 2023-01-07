# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

# [실습]
# 1. R2를 음수가 아닌 0.5 이하로 줄이기
# 2. 데이터는 건들지 말것
# 3. 레이어는 인풋 아웃풋 포함 7개 이상
# 4. batch_size = 1
# 5. 히든레이어의 노드는 각각 10개 이상 100개 이하
# 6. train 70%
# 7. epoch 100번 이상
# 8. loss 지표는 mse or mae
# 9. activation 사용 금지

import numpy as np                                                                  # 넘파이를 임포트시키고 간단하게 np로 명시해준다.
from tensorflow.keras.models import Sequential                                      # Sequential은 레이어에 순차적으로 연산
from tensorflow.keras.layers import Dense                                           # Dense는 완전 연결층을 구현하는 레이어 모델
from sklearn.model_selection import train_test_split                                # 사이킷런(scikit-learn)의 model_selection 패키지 안에 train_test_split 모듈을 활용하여 손쉽게 train set(학습 데이터 셋)과 test set(테스트 셋)을 분리할 수 있다.
from sklearn.metrics import mean_squared_error, r2_score                            # RMSE 함수는 아직 없어서 직접 만들어 사용. - 회귀 분석 모델 / 사이킷런에서도 rmse는 제공하지 않음. / MSE 함수 불러옴.
                                                                                    # MSE보다 이상치에 덜 민감하다. 이상치에 대한 민감도가 MSE보단 적고 MAE보단 크기 때문에 이상치를 적절히 잘 다룬다고 간주되는 경향이 있다고 한다.

x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(                                # 파라미터 (x와 y에 값을 대입) 
    x, y, train_size=0.7, random_state=123)                                         # 데이터셋에서 70%(x_train) # 데이터셋에서 30%(x_test) / 둘 중 하나만 명시해주면 된다. 
                                                                                    # test_size default=0.25
                                                                                    # shuffle=True 무작위 추출, False=순차적 추출 / default=True
                                                                                    # random_state로 잡아주면 그다음 데이터도 동일한 데이터로 들어감. 아무런 의미 없는 값을 넣어도 상관없다.

#2. 모델구성
model=Sequential()                                                                  # 모델은 순차적으로 구성하겠다.
model.add(Dense(400, input_dim=1))                                                  # 1열이라서 iuput_dim은 1       # x열의 개수
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))                                                                # Dense값을 임의로 조절
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(1))                                                                 # output=1                     # y열의 개수

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])                 # MSE는 추정된 값과 실제 값 간의 평균 제곱 차이를 의미한다. #주로 회귀에서 사용되는 손실 함수이며, 정확도 개념은 회귀에 적용되지 않는다고 한다.
                                                                                    # 일반적인 회귀 지표는 MAE이며, MSE는 손실 함수로 쓰이고 MAE는 회귀지표로써 사용된다.
                                                                                    # mae = mean(평균),absolute(절대값),error / mse = mean loss mse(평균제곱오차)
                                                                                    # loss : 훈련에 영향을 미친다. loss는 다음 가중치에 반영 → 반복 훈련
                                                                                    
                                                                                    # metrics = 어떤 방식으로 모델을 돌릴 것인가? 즉, loss = 손실함수, metrics = 평가지표
                                                                                    # metrics에 사용하는 ['mae', 'mse', 'accuracy', 'acc']는 훈련에 영향을 미치지 않는다. 참고용으로 사용하는데, 사용방법은 mae 또는 mse 아니면 ['mae', 'mse'] 이렇게 여러개의 리스트를 써도 된다.
                                                                                    # 'accuracy' = 'acc' / accuracy는 여기서 쓸 수 없는 평가지표이다.
                                                            
model.fit(x_train, y_train, epochs=100, batch_size=32)                              # epoch: 훈련을 여러번 시키는것. fit: 훈련시킴. # epoch 너무 많이하면 오히려 성능이 떨어지는 경우가 있기 때문에, 적절한 epoch 수치를 찾아내야 함. # 이번엔 훈련 수치를 100번으로 조정.
                                                                                    # batch_size가 크면 속도가 빨라지고 정확도가 떨어짐. 과적합(overfitting)을 고려해서 조절해야 함. # batch size default=32(기본값) # 여기서는 기본값으로 임의로 세팅

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
model.add(Dense(400, input_dim=1))                                                  # 1열이라서 iuput_dim은 1       # x열의 개수
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))                                                                # Dense값을 임의로 조절
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(1))                                                                 # output=1                     # y열의 개수

model.fit(x_train, y_train, epochs=100, batch_size=32)                              # epoch: 훈련을 여러번 시키는것. fit: 훈련시킴. # epoch 너무 많이하면 오히려 성능이 떨어지는 경우가 있기 때문에, 적절한 epoch 수치를 찾아내야 함. # 이번엔 훈련 수치를 100번으로 조정.
                                                                                    # batch_size가 크면 속도가 빨라지고 정확도가 떨어짐. 과적합(overfitting)을 고려해서 조절해야 함. # batch size default=32(기본값) # 여기서는 기본값으로 임의로 세팅

레이어를 마구마구 늘려주고, Dense값을 뒤죽박죽으로 만들어버려야징!!!

RMSE: 5.827593424223314,  R2: 0.19407355026379103

'''