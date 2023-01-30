# 52_2 복붙

import numpy as np

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x1_datasets.shape)                                                        # (100, 2) -> 삼전 시가+고가
print(x1_datasets)

x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose()
print(x2_datasets.shape)                                                        # (100, 3) -> 아모레 시가+고가+종가

x3_datasets = np.array([range(100, 200), range(1301, 1401)]).transpose()

y1 = np.array(range(2001, 2101))                                                 # (100,) -> 삼전 하루 뒤 '종가'
y2 = np.array(range(201, 301))                                                   # (100,) -> 아모레 하루 뒤 '종가'

# 실습!!! 만들어!!!

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, \
    x3_train, x3_test, y1_train, y1_test, \
    y2_train, y2_test = train_test_split(
        x1_datasets, x2_datasets, x3_datasets, 
        y1, y2, train_size=0.7, random_state=1234
    )

print(x1_train.shape, x2_train.shape, x3_train.shape, y1_train.shape, y2_train.shape)                            # (70, 2) (70, 3) (70, 2) (70,) (70,)
print(x1_test.shape, x2_test.shape, x3_test.shape, y1_test.shape, y2_test.shape)                                 # (30, 2) (30, 3) (30, 2) (30,) (30,)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1.
input1 = Input(shape=(2, ))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

#2-2 모델2.
input2 = Input(shape=(3, ))
dense21 = Dense(21, activation='linear', name='ds21')(input2)
dense22 = Dense(22, activation='linear', name='ds22')(dense21)
output2 = Dense(23, activation='linear', name='ds23')(dense22)

#2-3 모델3.
input3 = Input(shape=(2, ))
dense31 = Dense(11, activation='relu', name='ds31')(input3)
dense32 = Dense(12, activation='relu', name='ds32')(dense31)
dense33 = Dense(13, activation='relu', name='ds33')(dense32)
output3 = Dense(14, activation='relu', name='ds34')(dense33)

#2-4 모델병합
from tensorflow.keras.layers import Concatenate
merge1 = Concatenate()([output1, output2, output3])                    # dense4와 dense23이 인풋이고 이름을 mg1으로 정의하겠다.
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, name='mg3')(merge2)
last_output = Dense(1, name='last1')(merge3)                                         # y가 컬럼이 1개이므로 아웃풋은 1개.

# merge11 = concatenate([output1, output2, output3], name='mg11')                                # dense4와 dense23이 인풋이고 이름을 mg1으로 정의하겠다.
# merge12 = Dense(15, activation='relu', name='mg12')(merge11)
# merge13 = Dense(16, name='mg13')(merge12)
# last_output2 = Dense(1, name='last2')(merge13)                                         # y가 컬럼이 1개이므로 아웃풋은 1개.

#2-5. 모델5 분기1
dense5 = Dense(32, activation='relu', name='ds51')(last_output)
dense5 = Dense(32, activation='relu', name='ds52')(dense5)
output5 = Dense(33, activation='relu', name='ds53')(dense5)

#2.6 모델5 분기2
dense6 = Dense(32, activation='relu', name='ds61')(last_output)
dense6 = Dense(32, activation='relu', name='ds62')(dense6)
output6 = Dense(33, activation='relu', name='ds63')(dense6)

model = Model(inputs=[input1, input2, input3], outputs=[output5, output6])
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=10, batch_size=8)

#4. 평가 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print("Loss: ", loss)

# loss가 왜 값이 3개가 나올까???
# 출력되는 모델이 2개, 전체 모델 1개 해서 총 3개로 나온다!!!

"""  
ens2                                                                                                                                                                      
Loss:  5443.61376953125

ens3 / concatenate
Loss:  [12353.111328125, 12346.1396484375, 6.971975326538086]

ens3 / Concatenate
Loss:  [1846776.375, 1814666.375, 32109.955078125]

metrics=['mae']
[1648401.375, 1618070.625, 30330.810546875, 889.4013671875, 132.48228454589844]

"""