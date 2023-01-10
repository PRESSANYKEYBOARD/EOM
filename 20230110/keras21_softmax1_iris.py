from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_iris()                                          # load_iris 불러오기
print(datasets.DESCR)                                           # x 컬럼 4개, y 컬럼 1개 안에 3개의 클래스가 있다.
                                                                # 판다스.describe / .info()
print(datasets.feature_names)                                   # 판다스.columns

x = datasets.data                                               # 데이터 분리 / 50행 4열
y = datasets['target']
# print(x)
# print(y)                                                        # 0, 1, 2가 골고루 퍼져 있다.
# print(x.shape, y.shape)                                         # (150, 4) (150,)


'''
from sklearn.preprocessing import OneHotEncoder                     # 사이킷런에서 제공하는 원 핫 인코딩을 사용했다.

onehot = OneHotEncoder()
onehot.fit(y.reshape(-1, 1))                                        # reshape(-1, 1)은 우선 2차원 배열로 변경하라는 것을 의미한다.
y = onehot.transform(y.reshape(-1, 1)).toarray()

'''

x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, 
        random_state=333, 
        test_size=0.2,
        stratify=y 
)                                                               # train은 대략 120개 정도
                                                                # False의 문제점은...? 
                                                                # y_test가 전부 2이다. 2 제외하고 전부 out된다. 그러면 성능저하가 일어난다. (1을 예측못함)
                                                                # 분류모델에서 가장 치명적이다.
                                                                
                                                                # True의 문제점은...?
                                                                # 특정 클래스에서 배제하는 결과가 나올 수 있다. / 데이터의 균형 자체가 틀어질 수 있다. 따라서 데이터의 비율을 비슷하게 맞춰줘야 한다.
                                                                # stratify=y / 분류형 데이터일 경우에만 가능하다.

# 원핫-인코딩을 해봐!!!

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


'''
from keras.utils.np_utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

'''

# print(y_train)
# print(y_test)
print(x.shape, y.shape)                                         # (150, 4) (150, 3)            



#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(4, )))                         
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='linear'))
model.add(Dense(3, activation='softmax'))                       # y 종류의 갯수 / 클래수 갯수, y 클래스가 3개이면 3개이다.
                                                                # 각각의 확률을 책정한다.
                                                                # 다중분류일 때, 최종 output는 무조건 softmax 100%다!!!
                                                                # 모든 확률 다 더하면 1이다.
                                                                
                                                                # 수치화 했을때 조심해야 할 점은? 0, 1, 2 동일한 관계이기 때문에 동등한 조건을 만들어줘야 한다.
                                                                
                                                                # One-Hot Encoding(원-핫 인코딩) / 0, 1, 2 
                                                                # 좌표 형태로 만든다.
                                                                # scikit-learn에서 제공하는 머신러닝 알고리즘은 문자열 값을 입력 값으로 허락하지 않기 때문에 모든 문자열 값들을 숫자형으로 인코딩하는 전처리 작업(Preprocessing) 후에 머신러닝 모델에 학습을 시켜야 한다.
                                                                
                                                                
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',                # 다중 분류 손실함수를 categorical_crossentropy로 많이 사용한다. / 원 핫 인코딩을 거치게 된 후 사용.
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=1,
          validation_split=0.2,
          verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)

print(y_test[:5])
y_predict = model.predict(x_test[:10])                          # 3개의 값들을 더하면 각각 1이 된다.
print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)                            # argmax: 가장 큰 값을 나타낸다. / axis=1 행에서 있는 값들을 빼낸다.
print("y_pred(예측값): ", y_predict)                                # 예측값 출력
y_test = np.argmax(y_test, axis=1)
print("y_test(원래값): ", y_test)                                   # 원값 출력

acc = accuracy_score(y_test, y_predict)
print(acc)

