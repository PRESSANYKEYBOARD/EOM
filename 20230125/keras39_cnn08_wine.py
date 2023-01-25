# [실습]
#1. TRAIN 0.7 이상
#2. R2: 0.8 이상 / RMSE 사용

from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model, load_model        # model은 인풋 레이어를 명시해줘야함.
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import r2_score

path = './_save/'
# path = '.._save/'
# path = 'c:/study/_save/'                                               # 셋 다 모두 동일함.

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

# print("최소값:", np.min(x))                                           # 0
# print("최대값:", np.max(x))                                           # 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    random_state=333,
    test_size=0.2
    # stratify=y
)

scaler = MinMaxScaler()                                                 # x_train에 대한 범위의 가중치 생성
x_train = scaler.fit_transform(x_train)                     
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)                                      # (142, 13) (36, 13)

x_train = x_train.reshape(142, 13, 1, 1)
x_test = x_test.reshape(36, 13, 1, 1)
print(x_train.shape, x_test.shape)                                      # (142, 13, 1, 1) (36, 13, 1, 1)

# 2. 모델구성(순차형)
model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(13, 1, 1)))
model.add(Dense(25, activation='relu'))
model.add(Flatten())
model.add(Dense(15))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))                                                     
model.add(Dense(1, activation='linear'))
model.summary()                                                         # 노드의 총 갯수는 drop을 해도 모두 동일하다.
                                                                        # drop을 해도 평가할 때는 전체 노드는 다 쓴다. / 훈련시에만 

# 2. 모델구성(함수형)
# input1 = Input(shape=(30, ))                                            # 인풋레이어를 13 레이어에 준다
# Dense1 = Dense(50, activation='relu')(input1)
# drop1 = Dropout(0.5)(Dense1)
# Dense2 = Dense(40, activation='sigmoid')(drop1)
# drop2 = Dropout(0.3)(Dense2)
# Dense3 = Dense(30, activation='relu')(drop2)
# drop3 = Dropout(0.2)(Dense3)
# Dense4 = Dense(20, activation='linear')(Dense3)
# output1 = Dense(1, activation='linear')(Dense4)
# model = Model(inputs=input1, outputs=output1)                           # 시작하는 부분과 끝 부분이 어디인지 알려주는 부분
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',     
             metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                            #   restore_best_weights=False,
                              verbose=1, )

import datetime                                                                 # 데이터 타임 임포트해서 명시해준다.
date = datetime.datetime.now()                                                  # 현재 날짜와 시간이 date로 반환된다.

print(date)                                                                     # 2023-01-12 14:57:54.345489
print(type(date))                                                               # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")                                               # date를 str(문자형)으로 바꾼다.
                                                                                # 0112_1457
print(date)
print(type(date))                                                               # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'                                    # -는 연산 -가 아니라 글자이다. str형이기 때문에...
                                                                                # 0037-0.0048.hdf5

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                    #   filepath=path + 'MCP/keras31_ModelCheckPoint3.hdf5')
                      filepath = filepath + 'k39_08' + date + '_' + filename
)

                                                                                       
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)                                                    # val_loss 즉, 검증할 때 손실값이 출력된다.
                                                                        # 기준을 잡을 때, val_loss로 기준을 잡는다.
                                                                        
# model.save(path + "keras31_ModelCheckPoint3_save_model.h5")
                                                                    
#4. 평가, 예측
print("=============== 1. 기본 출력 ========================")
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse)

y_predict = model.predict(x_test)
print("y_test(원래값):", y_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어:", r2)


'''
mse: 0.029193008318543434
r2스코어: 0.8668331469505023

cnn result
r2스코어: 0.9020558063090791

'''