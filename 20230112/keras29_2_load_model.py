# [실습]
#1. TRAIN 0.7 이상
#2. R2: 0.8 이상 / RMSE 사용

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model                           # model은 인풋 레이어를 명시해줘야함.
                                                                                            # load model를 쓸 때 load model을 임포트
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
scaler.fit(x_train)                                           # x_train에 대한 범위의 가중치 생성
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x_train = scaler.fit_transform(x_train)                     # 31.32번 라인의 내용을 한 줄로 정리

# 2. 모델구성(함수형)
path = './_save'
# path = '.._save'
# path = 'c:/study/_save'                                               # 셋 다 모두 동일함.

# model.save(path + 'keras29_1_save_model.h5')
# model.save('./_save/keras29_1_save_model.h5')                             # 둘 중 하나를 써도 된다.

model = load_model(path + 'keras29_1_save_model.h5')
model.summary()                                                             # 함수형으로 하면 인풋 레이어가 보인다.

#3. 컴파일, 훈련
import time                                                                             # 시간을 임포트
model.compile(loss='mse', optimizer='adam',     
             metrics=['mae'])
                         
start = time.time()                                                                     # 시작 시간
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          verbose=1)                                                  # val_loss 즉, 검증할 때 손실값이 출력된다.
                                                                      # 기준을 잡을 때, val_loss로 기준을 잡는다.   
end = time.time()                                                     # 종료 시간

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse:', mse)
print('mae:', mae)

y_predict = model.predict(x_test)

print("y_test(원래값):", y_test)
r2 = r2_score(y_test, y_predict)
print(r2)

print("걸린시간 :", end - start)



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