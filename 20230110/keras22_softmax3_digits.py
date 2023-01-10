import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

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

# 원핫-인코딩을 해봐!!!

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train)
# print(y_test)
print(x.shape, y.shape)                                         # (1797, 64) (1797, 10)

#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(64, )))                         
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(30, activation='linear'))
model.add(Dense(10, activation='softmax'))                       # y 종류의 갯수 / 클래수 갯수, y 클래스가 3개이면 3개이다.
                                                                # 각각의 확률을 책정한다.
                                                                # 다중분류일 때, 최종 output는 무조건 softmax 100%다!!!
                                                                # 모든 확률 다 더하면 1이다.
                                                                
                                                                # 수치화 했을때 조심해야 할 점은? 0, 1, 2 동일한 관계이기 때문에 동등한 조건을 만들어줘야 한다.
                                                                
                                                                # One-Hot Encoding(원-핫 인코딩) / 0, 1, 2 
                                                                # 좌표 형태로 만든다.
                                                                # scikit-learn에서 제공하는 머신러닝 알고리즘은 문자열 값을 입력 값으로 허락하지 않기 때문에 모든 문자열 값들을 숫자형으로 인코딩하는 전처리 작업(Preprocessing) 후에 머신러닝 모델에 학습을 시켜야 한다.
                                                                
                                                                
#3. 컴파일, 훈련
import time
start = time.time()
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping            # 뭔가 호출한다. / 대문자로 되어있는 걸로 봐선 클래스 함수다.
EarlyStopping = EarlyStopping(monitor='val_loss', mode='min',   # 파이썬에서는 대/소문자 구별함. / val_loss를 기준으로 모니터링 할 것이다.
            patience=10, 
            restore_best_weights=True,
            verbose=1)                                          # mode= 갱신 / min(최소), max(최대), auto(자동)
                                                                # patience= 참을성 / val_loss를 기준으로 최소값 대비 10번 참고 일찍 끝내겠다.
                                                                # patience는 default 끊은 시점에서 w값이 저장된다.
                                                                # restore_best_weights default = false / 최소의 loss와 최적의 w값을 반환하기 위해서 옵션을 true로 지정한다.
                                                                # verbose= 1 

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.2,
          verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)

print(y_test[:5])
y_predict = model.predict(x_test[:10])                              # 3개의 값들을 더하면 각각 1이 된다.
print(y_predict)


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)                            # argmax: 가장 큰 값을 나타낸다. / axis=1 행에서 있는 값들을 빼낸다.
print("y_pred(예측값): ", y_predict)                                # 예측값 출력
y_test = np.argmax(y_test, axis=1)
print("y_test(원래값): ", y_test)                                   # 원값 출력

acc = accuracy_score(y_test, y_predict)
print(acc)                       

import matplotlib.pyplot as plt
plt.gray()                                              # 1700 x 가로 8 x 세로 8 x 흑백 1
                                                        # 행 1700 x 열 64
                                                        # input.shape(64, ) y 값이 10개 (0 ~ 9)
plt.matshow(datasets.images[4])
plt.show()

'''
CPU
acc: 0.8777777777777778

GPU
acc: 0.9333333333333333

'''
