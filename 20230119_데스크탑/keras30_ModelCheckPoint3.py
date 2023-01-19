# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

# [실습]
#1. TRAIN 0.7 이상
#2. R2: 0.8 이상 / RMSE 사용

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model                           # model은 인풋 레이어를 명시해줘야함.
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_save/'
# path = '.._save'
# path = 'c:/study/_save'                                                                   # 셋 다 모두 동일함.

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

# print("최소값:", np.min(x))                             # 0
# print("최대값:", np.max(x))                             # 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    random_state=333,
    test_size=0.2
    # stratify=y
)

scaler = MinMaxScaler()                                      # x_train에 대한 범위의 가중치 생성
x_train = scaler.fit_transform(x_train)                     
x_test = scaler.transform(x_test)


#2. 모델구성(순차형)
# model = Sequential()
# model.add(Dense(50, activation='relu', input_shape=(13,)))
# model.add(Dense(40, activation='sigmoid'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='linear'))
# model.add(Dense(1, activation='linear'))
# model.summary()
# # Total params: 4,611                                                                                 # 총 연산량: 4611

# 2. 모델구성(함수형)
input1 = Input(shape=(13, ))                                                                            # 인풋레이어를 13 레이어에 준다
Dense1 = Dense(50, activation='relu')(input1)
Dense2 = Dense(40, activation='sigmoid')(Dense1)
Dense3 = Dense(30, activation='relu')(Dense2)
Dense4 = Dense(20, activation='linear')(Dense3)
output1 = Dense(1, activation='linear')(Dense4)
model = Model(inputs=input1, outputs=output1)                                                           # 시작하는 부분과 끝 부분이 어디인지 알려주는 부분
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',     
             metrics=['mae'])                                                                           # 모델을 더 이상 학습을 못할 경우(loss, metric등의 개선이 없을 경우), 학습 도중 미리 학습을 종료시키는 콜백함수       

es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                              verbose=1, restore_best_weights=True)                                     # 모델을 저장할 때 사용되는 콜백함수

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,  
                      filepath=path + 'MCP/keras30_ModelCheckPoint3.hdf5')                              # 저장 포인트
                                                                                                        # 시작 시간
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)                                                                                    # val_loss 즉, 검증할 때 손실값이 출력된다.
                                                                                                        # 기준을 잡을 때, val_loss로 기준을 잡는다.   
                                                                    
model.save(path + 'keras30_ModelCheckPoint3_save_model.h5')                                                            # 결과치는 0.8175761021927634
# model.save('./_save/keras29_3_save_model.h5')                                                         # 둘 중 하나를 써도 된다.
                                                                                                        # 모델 저장 (가중치 포함 안됨)
                                                                                                        
                                                                    
#4. 평가, 예측

print('========================1. 기본 출력============================')
mse, mae = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('1. mse : ' , mse)
print('1. R2_스코어 : ', r2)


print('========================2. load_model 출력============================')
model2 = load_model(path + 'keras30_ModelCheckPoint3_save_model.h5')
mse, mae = model2.evaluate(x_test, y_test)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('2. mse : ' , mse)
print('2. R2_스코어 : ', r2)


print('========================3. ModelCheckPoint 출력============================')
#가장 좋은 지점을 가지고만 사용했기 때문에 제일 성능 좋게 나온다.
#선생님은 CheckPoint 사용
model3 = load_model(path + 'MCP/keras30_ModelCheckPoint3.hdf5')
mse, mae = model3.evaluate(x_test, y_test)
print('3. mse : ' , mse)

y_predict = model3.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('3. R2_스코어 : ', r2)


'''
MCP 저장: 0.80625425220844280

'''
"""
CPU
=======================1. 기본 출력============================
4/4 [==============================] - 0s 665us/step - loss: 16.3871 - mae: 2.6684
1. mse :  16.387060165405273
1. R2_스코어 :  0.8329197537548421
========================2. load_model 출력============================
4/4 [==============================] - 0s 998us/step - loss: 16.3871 - mae: 2.6684
2. mse :  16.387060165405273
2. R2_스코어 :  0.8329197537548421
========================3. ModelCheckPoint 출력============================
4/4 [==============================] - 0s 666us/step - loss: 16.3871 - mae: 2.6684
3. mse :  16.387060165405273
3. R2_스코어 :  0.8329197537548421

GPU
========================1. 기본 출력============================
4/4 [==============================] - 0s 3ms/step - loss: 18.1037 - mae: 2.7101
1. mse :  18.103742599487305
1. R2_스코어 :  0.8154166668642767
========================2. load_model 출력============================
4/4 [==============================] - 0s 2ms/step - loss: 18.1037 - mae: 2.7101
2. mse :  18.103742599487305
2. R2_스코어 :  0.8154166668642767
========================3. ModelCheckPoint 출력============================
4/4 [==============================] - 0s 2ms/step - loss: 18.1037 - mae: 2.7101
3. mse :  18.103742599487305
3. R2_스코어 :  0.8154166668642767

"""

"""
'''
epochs=1000
Epoch 00001: val_loss improved from inf to 522.52173, saving model to ./_save/MCP\keras30_ModelCheckPoint1.hdf5
-> 처음 훈련은 최상의 결과값이므로 저장
Epoch 00002: val_loss improved from 522.52173 to 444.32184, saving model to ./_save/MCP\keras30_ModelCheckPoint1.hdf5
-> 2번째 훈련 개선 -> 덮어쓰기
-> 반복

Epoch 00041: val_loss did not improve from 9.04129
-> 개선되지 않을 경우 저장 X
-> 개선되지 않은 결과가 20번 반복될 경우, EarlyStopping = 가장 성능이 좋은 ModelCheckPoint 지점

Epoch 81/1000
1/9 [==>...........................] - ETA: 0s - loss: 6.8561 - mae: 1.6694
Restoring model weights from the end of the best epoch: 61.
EarlyStopping: 최적의 weight가 갱신이 안되면 훈련을 끊어주는 역할
ModelCheckPoint: 최적의 weight가 갱신될 때마다 저장해주는 역할

MCP 저장
RMSE:  4.393303432855621
R2:  0.7612078343831213
-> 확인하는 파일: keras30_ModelCheckPoint2_load_model.py

"""