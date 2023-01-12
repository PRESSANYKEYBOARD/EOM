# [실습]
#1. TRAIN 0.7 이상
#2. R2: 0.8 이상 / RMSE 사용

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model        # model은 인풋 레이어를 명시해줘야함.
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.metrics import r2_score

path = './_save/'
# path = '.._save/'
# path = 'c:/study/_save/'                                               # 셋 다 모두 동일함.

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

"""
#2. 모델구성(순차형)
# model = Sequential()
# model.add(Dense(50, activation='relu', input_shape=(13,)))
# model.add(Dense(40, activation='sigmoid'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='linear'))
# model.add(Dense(1, activation='linear'))
# model.summary()
# # Total params: 4,611                                              # 총 연산량: 4611

# 2. 모델구성(함수형)
input1 = Input(shape=(13, ))                                     # 인풋레이어를 13 레이어에 준다
Dense1 = Dense(50, activation='relu')(input1)
Dense2 = Dense(40, activation='sigmoid')(Dense1)
Dense3 = Dense(30, activation='relu')(Dense2)
Dense4 = Dense(20, activation='linear')(Dense3)
output1 = Dense(1, activation='linear')(Dense4)
model = Model(inputs=input1, outputs=output1)                    # 시작하는 부분과 끝 부분이 어디인지 알려주는 부분
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',     
             metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                              verbose=1, restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                      filepath=path + 'MCP/keras30_ModelCheckPoint1.hdf5')
                                                                                       # 시작 시간
model.fit(x_train, y_train, epochs=5000, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)                                                    # val_loss 즉, 검증할 때 손실값이 출력된다.
                                                                        # 기준을 잡을 때, val_loss로 기준을 잡는다.
                                                                        
                                                                              
                                                                    
# model.save(path + 'keras29_3_save_model.h5')                            # 결과치는 0.8175761021927634
# # model.save('./_save/keras29_3_save_model.h5')                         # 둘 중 하나를 써도 된다.

"""

model = load_model(path + 'MCP/keras30_ModelCheckPoint1.hdf5')
                                                                    
#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse)
print('mae:', mae)

y_predict = model.predict(x_test)

print("y_test(원래값):", y_test)
r2 = r2_score(y_test, y_predict)
print(r2)

'''
MCP 저장: 0.8062542522084428
0.8119358397129732


'''