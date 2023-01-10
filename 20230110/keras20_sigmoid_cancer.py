import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.  데이터
datasets = load_breast_cancer()
# print(datasets)                         # x 값은 많은 데이터, y 값은 0과 1로 구성되어 있음.
#                                         # target_names: 암에 걸렸다 / 안 걸렸다.
#                                         # 


# print(datasets.DESCR)
# print(datasets.feature_names)           # 30개

'''
    radius (mean):                        6.981  28.11
    texture (mean):                       9.71   39.28
    perimeter (mean):                     43.79  188.5
    area (mean):                          143.5  2501.0
    smoothness (mean):                    0.053  0.163
    compactness (mean):                   0.019  0.345
    concavity (mean):                     0.0    0.427
    concave points (mean):                0.0    0.201
    symmetry (mean):                      0.106  0.304
    fractal dimension (mean):             0.05   0.097
    radius (standard error):              0.112  2.873
    texture (standard error):             0.36   4.885
    perimeter (standard error):           0.757  21.98
    area (standard error):                6.802  542.2
    smoothness (standard error):          0.002  0.031
    compactness (standard error):         0.002  0.135
    concavity (standard error):           0.0    0.396
    concave points (standard error):      0.0    0.053
    symmetry (standard error):            0.008  0.079
    fractal dimension (standard error):   0.001  0.03
    radius (worst):                       7.93   36.04
    texture (worst):                      12.02  49.54
    perimeter (worst):                    50.41  251.2
    area (worst):                         185.2  4254.0
    smoothness (worst):                   0.071  0.223
    compactness (worst):                  0.027  1.058
    concavity (worst):                    0.0    1.252
    concave points (worst):               0.0    0.291
    symmetry (worst):                     0.156  0.664
    fractal dimension (worst):            0.055  0.208

'''
                                
x = datasets['data']                  # datasets.data와 같음
y = datasets['target']                # datasets.target 같음
# print(x.shape, y.shape)               # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, test_size=0.2 
)                                                           # train은 대략 404개 정도

#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='linear', input_shape=(30,)))            # input_dim=30
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))                               # 0과 1 사이의 값을 찾아야 되기 때문에 sigmoid로 맞춰준다.
                                                                        # 2진 분류네???

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',             # binary_crossentropy = 2진 분류에서 사용하는 손실 함수
              metrics=['accuracy'])                                     # 

from tensorflow.keras.callbacks import EarlyStopping            # 뭔가 호출한다. / 대문자로 되어있는 걸로 봐선 클래스 함수다.
EarlyStopping = EarlyStopping(monitor='val_loss', mode='min',   # 파이썬에서는 대/소문자 구별함. / val_loss를 기준으로 모니터링 할 것이다.
            patience=10, 
            restore_best_weights=True,
            verbose=1)                                          # mode= 갱신 / min(최소), max(최대), auto(자동)
                                                                # patience= 참을성 / val_loss를 기준으로 최소값 대비 10번 참고 일찍 끝내겠다.
                                                                # patience는 default 끊은 시점에서 w값이 저장된다.
                                                                # restore_best_weights default = false / 최소의 loss와 최적의 w값을 반환하기 위해서 옵션을 true로 지정한다.
                                                                # verbose= 1 

model.fit(x_train, y_train, epochs=10000, batch_size=16,
          validation_split=0.2,
          callbacks=[EarlyStopping],
          verbose=1)

#4. 평가 ,예측
loss = model.evaluate(x_test, y_test)
print('loss, accuracy:', loss)
loss, accuracy = model.evaluate(x_test, y_test)                 # 첫번재는 loss, 두번재는 accuracy로 반환
print('loss:', loss)
print('accuracy:', accuracy)

# → 정수형으로 바꾸자!!!

y_predict = model.predict(x_test)
y_predict = np.asarray(y_predict, dtype = int)                      # 내가 찾은 답인데, 맞는건지 확실하지가 않음. / 반올림 되어있지 않은 형태이다.
                                                                    # np.asarray: 입력된 데이터를 np.array 형식으로 만듬. 
                                                                    # dtype 속성: 데이터 형식 변경 / (int: 정수형 / float: 실수형 / complex: 복소수형 / str: 문자형)

# y_predict = y_predict.astype('int')                               # 위에 있는 내용과 출력값이 동일.
                                                                    # astype: y_predict의 데이터형 변환(int형으로!)

# y_predict = (list(map(int, y_predict)))                           # 깃허브 사찰해서 나온 답안
                                                                    # map: 리스트의 요소를 지정된 함수로 처리 / list(map(함수, 리스트))

print(y_predict[:10])                                           # 실수형으로 출력된 값 
print(y_test[:10])                                              # 정수로 출력된 값

from sklearn.metrics import r2_score, accuracy_score    
acc = accuracy_score(y_test, y_predict)
print("accuracy_score:", acc)                                   # 그냥 실행하면 오류 / 실수형과 정수형과의 매칭이 안 되니 오류가 발생
                                                                # 

'''
binary_crossentropy: 0.1934686005115509
accuracy: 



'''