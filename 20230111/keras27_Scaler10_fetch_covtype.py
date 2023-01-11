# 완성시켜라

import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)                             # (581012, 54) (581012, )
print(np.unique(y, return_counts=True))             
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#   dtype=int64))

'''
#1. to_categorical() 사용 / 케라스 투카테고리컬
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
y = np.delete(y, 0, axis=1)                                     # y의 0번째 열을 제거한다.
print(y.shape)
                                                                # 8개 컬럼. 0번째 컬럼 제거 np.delete
                                                                # y_pred(예측값):  [1 0 1 ... 1 1 0]
                                                                # y_test(원래값):  [1 6 4 ... 1 1 0]
                                                                # 0.716737089403888
                                                                # Mission Complete!!!
 .
 .
 .
 
 y = to_categorical(y)                                          # 원 핫 인코딩
 print(y.shape)                                                 # (581012, 8)
 print(type(y))                                                 # <class 'numpy.ndarray'>
 print(y[:10])          
 print(np.unique(y[:,0], return_counts=True))                   
 # (array[0.], dtype=float32), array([581012], dtype=int64))
 
 print(np.unique(y[:,1], return_counts=True))
 print("=========================================")
 y = np.delete(y, 0, axis=1)
 print(y.shape)
 print(y[:10])
 print(np.unique(y[:,0], return_counts=True))
 
                                                                                                            
'''

'''
#2. get_dummies() 사용 / 판다스 겟더미스
import pandas as pd                                                     
y = pd.get_dummies(y)                                                   

print(type(y))                                                          # 형차이로 인해 오류발생
                                                                        # .values or .numpy() → 판다스에서 넘파이 형태로 바꾸는것
# y = y.values                                                          # y_pred(예측값):  [1 6 1 ... 1 1 0]
                                                                        # y_test(원래값):  [1 6 4 ... 1 1 0]
                                                                        # 0.7260741977401616
y = y.to_numpy()                                                        # Mission Complete!!!

.
.
.

import pandas as pd     
y = pd.get_dummies(y)
print(y[:10])
print(type(y))                                                          # 판다스의 데이터 형태는 인덱스와 헤더가 자동 생성된다.
                                                                        # <class 'pandas.core.frame.DataFrame'>
                                                                        # 넘파이 자료형이 판다스 형태를 못 받아들인다.
                                                                        # 텐서플로에서는 넘파이와 판다스 형태는 둘 다 받아들여서 연산까지 해준다.
                                                                        # y_test 형태가 판다스 형태이기 때문에 179번 라인 argmax 부분에서 에러 발생

# y = y.values                                                          # y가 넘파이 형태로 바뀌어진다.
print(type(y))                                                          # <class 'numpy.ndarray'>

y = y.to_numpy()                                                        # y가 넘파이 형태로 바뀌어진다.
print(type(y))                                                          # <class 'numpy.ndarray'>
print(y.shape)                                                          # (581012, 7)

'''                                                   

                                                          
#3. OneHotEncoder() 사용 / 사이킷런 원핫인코더                                                   
from sklearn.preprocessing import OneHotEncoder                 # 사이킷런에서 제공하는 원 핫 인코딩을 사용했다.

''' 

onehot = OneHotEncoder()                                        
onehot.fit(y.reshape(-1, 1))                                    # reshape: 행/열 재배치
                                                                # 벡터 입력을 허용하지 않음 → reshape을 이용해 Matrix로 변환 필요.
                                                                # 매트릭스: 2차원 행렬
y = onehot.transform(y.reshape(-1, 1)).toarray()                # transform(): 입력된 개체와 동일하게 인덱스 된 객체를 반환한다.

                                                                # .toarray
                                                                # y_pred(예측값):  [1 0 1 ... 1 1 0]
                                                                # y_test(원래값):  [1 6 4 ... 1 1 0]
                                                                # 0.7210571155650026
                                                                # Mission Complete!!!
                                                                
.
.
.

'''

print(y.shape)                                                  # (581012, )
y = y.reshape(581012, 1)
print(y.shape)                                                  # (581012, 1)

ohe = OneHotEncoder()                                           # 원 핫 인코더를 ohe라는 변수로 지정
                                                                # 1차원 배열로 줬다고 오류를 내뿜음.
                                                                # Reshape를 할 때 데이터의 내용과 순서가 변하지 않아야 한다!!!!!!!!!
# ohe.fit(y)
# y = ohe.transform(y)
y = ohe.fit_transform(y)
y = y.toarray()                                                 # numpy 형태로 바꾸자.
print(type(y))                                                  # <class 'numpy.ndarray'>

print(y[:15])
print(type(y))                                                  # <class 'scipy.sparse._csr.csr_matrix'>
print(y.shape)                                                  # (581012, 7)


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
                                                                
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)                                           # x_train에 대한 범위의 가중치 생성
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x_train = scaler.fit_transform(x_train)                     # 한 줄로 정리
                                                                
                                                              

# 원핫-인코딩을 해봐!!!

# print(y_train)
# print(y_test)
print(x.shape, y.shape)                                         # (581012, 54) (581012, 7)

print(datasets.DESCR)
print(datasets.feature_names)

#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(54, )))                         
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(30, activation='linear'))
model.add(Dense(7, activation='softmax'))                       # y 종류의 갯수 / 클래수 갯수, y 클래스가 3개이면 3개이다.
                                                                # 각각의 확률을 책정한다.
                                                                # 다중분류일 때, 최종 output는 무조건 softmax 100%다!!!
                                                                # 모든 확률 다 더하면 1이다.
                                                                
                                                                # 수치화 했을때 조심해야 할 점은? 0, 1, 2 동일한 관계이기 때문에 동등한 조건을 만들어줘야 한다.
                                                                
                                                                # One-Hot Encoding(원-핫 인코딩) / 0, 1, 2 
                                                                # 좌표 형태로 만든다.
                                                                # scikit-learn에서 제공하는 머신러닝 알고리즘은 문자열 값을 입력 값으로 허락하지 않기 때문에 모든 문자열 값들을 숫자형으로 인코딩하는 전처리 작업(Preprocessing) 후에 머신러닝 모델에 학습을 시켜야 한다.
                                                                
                                                                
#3. 컴파일, 훈련
import time                                                                             # 시간을 임포트
model.compile(loss='mse', optimizer='adam',                
             metrics=['mae'])
                         
start = time.time()                                                                     # 시작 시간
model.fit(x_train, y_train, epochs=10, batch_size=32,
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

MSE: 0.051108550280332565
MAE: 0.1007518321275711
걸린시간: 146.94920086860657

GPU

MSE: 0.04932910576462746
MAE: 0.09575667977333069
걸린시간: 132.2925283908844

...
...

StandardScaler

CPU

MSE: 0.046931833028793335
MAE: 0.09023147076368332
걸린시간: 139.87093544006348

GPU

MSE: 0.04715726152062416
MAE: 0.09198630601167679
걸린시간: 143.74699306488037

'''        