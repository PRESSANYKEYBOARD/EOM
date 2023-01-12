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

path = './_save/'
# path = '.._save'
# path = 'c:/study/_save'                                               # 셋 다 모두 동일함.

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


#  2. 모델구성(함수형)
input1 = Input(shape=(13, ))                                     # 인풋레이어를 13 레이어에 준다
Dense1 = Dense(50, activation='relu')(input1)
Dense2 = Dense(40, activation='sigmoid')(Dense1)
Dense3 = Dense(30, activation='relu')(Dense2)
Dense4 = Dense(20, activation='linear')(Dense3)
output1 = Dense(1, activation='linear')(Dense4)
model = Model(inputs=input1, outputs=output1)                    # 시작하는 부분과 끝 부분이 어디인지 알려주는 부분
model.summary()

# model.save_weights(path + 'keras29_5_save_weights1.h5')                               # 결과치는 0.8175761021927634
# model.save('./_save/keras29_5_save_weights.h5')                                       # 둘 중 하나를 써도 된다.

model.load_weights(path + 'keras29_5_save_weights1.h5')                                 # 가중치만 저장이 되며 모델에서 사용을 못한다.
                                                                                        # 모델이 미리 정의가 되어야 사용을 할 수 있다.
                                                                                        # 컴파일과 훈련이 되어있지 않은 상태에서 저장이 된 것이다.
                                                                                        # 


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',     
             metrics=['mae'])
                         
# model.fit(x_train, y_train, epochs=100, batch_size=32,
#           validation_split=0.2,
#           verbose=1)                                                    # val_loss 즉, 검증할 때 손실값이 출력된다.
#                                                                         # 기준을 잡을 때, val_loss로 기준을 잡는다.   
                                                                    
model.save_weights(path + 'keras29_5_save_weights2.h5')                             # 결과치는 0.8175761021927634
                                                                                    # 컴파일 안되어있다고 오류를 내뿜음.
                                                                                    # 모델과 컴파일까지 명시가 되어있어야 한다.
                                                                                    
# model.save_weights('./_save/keras29_5_save_weights2.h5')                          # 둘 중 하나를 써도 된다.                                  

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse)
print('mae:', mae)

y_predict = model.predict(x_test)

print("y_test(원래값):", y_test)
r2 = r2_score(y_test, y_predict)
print(r2)


'''
MinMaXScaler

CPU

mse: 18.270593643188477
mae: 2.693286895751953
걸린시간 : 3.9115521907806396

GPU

mse: 18.068389892578125
mae: 2.700852870941162
걸린시간: 3.7869598865509033

...
...

StandardScaler

CPU

mse: 17.717557907104492
mae: 2.6360278129577637
걸린시간: 3.8994944095611572

GPU

mse: 16.36928367614746
mae: 2.386413335800171
걸린시간: 33.859226703643799

'''