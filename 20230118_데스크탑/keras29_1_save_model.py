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
from tensorflow.keras.models import Sequential, Model                           # model은 인풋 레이어를 명시해줘야함.
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import r2_score

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

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)                                                         # x_train에 대한 범위의 가중치 생성
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x_train = scaler.fit_transform(x_train)                                   # 31.32번 라인의 내용을 한 줄로 정리

#2. 모델구성(순차형)
# model = Sequential()
# model.add(Dense(50, activation='relu', input_shape=(13,)))
# model.add(Dense(40, activation='sigmoid'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='linear'))
# model.add(Dense(1, activation='linear'))
# model.summary()
# # Total params: 4,611                                                     # 총 연산량: 4611

# 2. 모델구성(함수형)
input1 = Input(shape=(13, ))                                                # 인풋레이어를 13 레이어에 준다
Dense1 = Dense(50, activation='relu')(input1)
Dense2 = Dense(64, activation='sigmoid')(Dense1)
Dense3 = Dense(32, activation='relu')(Dense2)
Dense4 = Dense(16, activation='linear')(Dense3)
output1 = Dense(1, activation='linear')(Dense4)
model = Model(inputs=input1, outputs=output1)                               # 시작하는 부분과 끝 부분이 어디인지 알려주는 부분
model.summary()


path = './_save/'
# path = '.._save/'
# path = 'c:/study/_save'                                                   # 셋 다 모두 동일함.

model.save(path + 'keras29_1_save_model.h5')                                # 모델 저장 (가중치 포함 안됨)
# model.save('./_save/keras29_1_save_model.h5')                             # 둘 중 하나를 써도 된다.