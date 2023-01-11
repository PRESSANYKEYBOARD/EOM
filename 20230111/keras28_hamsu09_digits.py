import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential, Model                           # model은 인풋 레이어를 명시해줘야함.
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)                                 # (1797, 64) (1797,)
print(np.unique(y, return_counts=True))                 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

print(datasets.DESCR)
#    :Number of Instances: 1797
#     :Number of Attributes: 64
#     :Attribute Information: 8x8 image of integer pixels in the range 0..16.
#     :Missing Attribute Values: None
#     :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
#     :Date: July; 1998


print(datasets.feature_names)
# ['pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'pixel_0_5', 'pixel_0_6', 'pixel_0_7', 'pixel_1_0', 'pixel_1_1', 'pixel_1_2', 
# 'pixel_1_3', 'pixel_1_4', 'pixel_1_5', 'pixel_1_6', 'pixel_1_7', 'pixel_2_0', 'pixel_2_1', 'pixel_2_2', 'pixel_2_3', 'pixel_2_4', 'pixel_2_5', 'pixel_2_6', 'pixel_2_7', 'pixel_3_0', 'pixel_3_1', 'pixel_3_2', 'pixel_3_3', 'pixel_3_4', 'pixel_3_5', 'pixel_3_6', 'pixel_3_7', 'pixel_4_0', 'pixel_4_1', 'pixel_4_2', 'pixel_4_3', 'pixel_4_4', 'pixel_4_5', 'pixel_4_6', 'pixel_4_7', 'pixel_5_0', 'pixel_5_1', 'pixel_5_2', 'pixel_5_3', 'pixel_5_4', 'pixel_5_5', 'pixel_5_6', 'pixel_5_7', 'pixel_6_0', 'pixel_6_1', 'pixel_6_2', 'pixel_6_3', 'pixel_6_4', 'pixel_6_5', 'pixel_6_6', 'pixel_6_7', 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixel_7_3', 'pixel_7_4', 'pixel_7_5', 'pixel_7_6', 'pixel_7_7']

x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, 
        random_state=333, 
        test_size=0.2,
        stratify=y 
)                                                               
                                                                # False의 문제점은...? 
                                                                # y_test가 전부 2이다. 2 제외하고 전부 out된다. 그러면 성능저하가 일어난다. (1을 예측못함)
                                                                # 분류모델에서 가장 치명적이다.
                                                                
                                                                # True의 문제점은...?
                                                                # 특정 클래스에서 배제하는 결과가 나올 수 있다. / 데이터의 균형 자체가 틀어질 수 있다. 따라서 데이터의 비율을 비슷하게 맞춰줘야 한다.
                                                                # stratify=y / 분류형 데이터일 경우에만 가능하다.
                                                                
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)                                           # x_train에 대한 범위의 가중치 생성
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x_train = scaler.fit_transform(x_train)                     # 한 줄로 정리

# 원핫-인코딩을 해봐!!!

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train)
# print(y_test)
print(x.shape, y.shape)                                         # (1797, 64) (1797, 10)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(5, activation='relu', input_shape=(64, )))                         
# model.add(Dense(50, activation='sigmoid'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(30, activation='linear'))
# model.add(Dense(10, activation='softmax'))                       # y 종류의 갯수 / 클래수 갯수, y 클래스가 3개이면 3개이다.
                                                                # 각각의 확률을 책정한다.
                                                                # 다중분류일 때, 최종 output는 무조건 softmax 100%다!!!
                                                                # 모든 확률 다 더하면 1이다.
                                                                
                                                                # 수치화 했을때 조심해야 할 점은? 0, 1, 2 동일한 관계이기 때문에 동등한 조건을 만들어줘야 한다.
                                                                
                                                                # One-Hot Encoding(원-핫 인코딩) / 0, 1, 2 
                                                                # 좌표 형태로 만든다.
                                                                # scikit-learn에서 제공하는 머신러닝 알고리즘은 문자열 값을 입력 값으로 허락하지 않기 때문에 모든 문자열 값들을 숫자형으로 인코딩하는 전처리 작업(Preprocessing) 후에 머신러닝 모델에 학습을 시켜야 한다.

# 2. 모델구성(함수형)
input1 = Input(shape=(64, ))                                     
Dense1 = Dense(50, activation='relu')(input1)
Dense2 = Dense(50, activation='relu')(Dense1)
Dense3 = Dense(64, activation='sigmoid')(Dense2)
Dense4 = Dense(32, activation='relu')(Dense3)
Dense5 = Dense(16, activation='linear')(Dense4)
output1 = Dense(10, activation='softmax')(Dense5)
model = Model(inputs=input1, outputs=output1)                    # 시작하는 부분과 끝 부분이 어디인지 알려주는 부분
model.summary()                                                              
                                                                
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

#4. 평가 ,예측
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

mse: 0.006445649545639753
mae: 0.00926679465919733
걸린시간: 7.257876396179199

GPU

mse: 0.006048947107046843
mae: 0.009545286186039448
걸린시간: 7.902857780456543

...
...

StandardScaler

CPU

mse: 0.004226148594170809
mae: 0.0076755681075155735
걸린시간: 7.2015767097473145

GPU

mse: 0.00322093372233212
mae: 0.0066462317481637
걸린시간: 7.598920106887817

'''            
