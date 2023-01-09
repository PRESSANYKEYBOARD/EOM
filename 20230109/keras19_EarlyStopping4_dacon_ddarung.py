import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    # 데이터에서 제외할 인덱스 칼럼이 0열에 있음. 열은 데이터가 아닌 인덱스라고 알려줘서 데이터로 들어가는 것을 방지.
                                                            # train_csv = pd.read_csv('./_data/ddarung/train.csv,' index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)  
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)                                      # (1459, 10) 카운트 분리하면 (1459, 9)

print(train_csv.columns)
                                                            # ['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                                                                #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                                                                #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],

print(train_csv.info())                                     # 결측치 데이터가 2개가 있음. 임의의 데이터를 넣으면 오차가 커지기 때문에 제거해야 한다. 단, 데이터가 적을시에는 삭제하는 것이 오히려 치명적이다.
print(test_csv.info())                                      # 카운트 없음.
print(train_csv.describe())

train_csv = train_csv.dropna()
print(train_csv.isnull().sum())                             # 결측값 확인
print(train_csv.shape)                                      # (1328, 10)

x = train_csv.drop(['count'], axis=1)                       # 작업할 때 axis가 1이면 행, 0이면 열
print(x)                                                    # [1459 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)                                              # (1459, )
                                                           
x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, test_size=0.2 
)                                                               # train은 대략 1167개 정도

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=9))                                # input_dim은 행과 열만 되어있는 곳만 쓸 수 있다.
model.add(Dense(40, input_shape=(9, )))                          # 스칼라가 9개로 볼 수 있기 때문에, (9, ) 라고 해석할 수 있다.
                                                                # (100, 10, 5) 라는 데이터가 들어가면, 행 무시 열 우선이기 때문에 (10, 5)로 들어감.
                                                                # 앞으로는 이렇게 쓰자!!!
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(35))
model.add(Dense(1))

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

hist = model.fit(x_train, y_train, epochs=300, batch_size=32,
          validation_split=0.2, callbacks=(EarlyStopping), 
          verbose=1)                                        # verbose= 진행표시줄 on/off
                                                            # default=1
                                                            # hist = history
                                                            # model.fit 값을 받아서 hist라는 변수로 리턴한다.
                                                            
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)

print("걸린시간:", end - start)
                                                            # verbose=1     진행바가 보임 / 모든게 다 나옴
                                                            # verbose=0     진행바 그거 뭐임? / 훈련 과정이 안 보임
                                                            # verbose=2     loss, metrics값만 나오고 진행바는 안 보임
                                                            # verbose=3     epoch만 진행됨
             
print("==============================")                                               
print(hist)     # <keras.callbacks.History object at 0x000001BF527C0AF0>
print("==============================")
print(hist.history)                                         # hist 공간 안에 history 라는 제공된 변수가 있다.
                                                            # loss와 val_loss 값이 딕셔너리 형태로 들어가 있다.
                                                            # 파이썬의 데이터 형태는 list / key / value
                                                            # dictionary(딕셔너리): 형태와 {} 형태로 묶여져 있는데, value 값이 list 형태로 묶여져 있다.
                                                            # 딕셔너리는 key와 value 형태 / 홍길동: {국영수 점수} / 심청이 :{국영수 점수}
                                                            # 두 개 이상=list 형태
                                                            
print("==============================")
print(hist.history['loss'])                                 # loss 값 출력
print("==============================")
print(hist.history['val_loss'])                             

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))                                   # 그림에 대한 판 사이즈
plt.plot(hist.history['loss'], c='red', 
         marker='.', label='loss')                          # list 형태에서 x 명시는 굳이 안 해도 상관 없다. y만 넣어주면 된다.
                                                            # c= 색깔 지정
                                                            # marker= 선의 모양
                                                            # label= 선의 이름
                                                
plt.plot(hist.history['val_loss'], c='blue', 
         marker='.', label='val_loss')              
               
plt.grid()                                                  # 격자 넣기
plt.xlabel('epochs')                                        # x축을 epochs 지정
plt.ylabel('loss')                                          # y축을 loss 지정
plt.title('ddarung loss')                                   # 제목을 지정
plt.legend()                                                # 라벨이 나오게 된다 / 그래프가 없는 지점에 나온다.
# plt.legend(loc='upper left')                              # loc= 라벨을 어디에 나오게? / upper right
plt.show()                                                  # 그림 보여주기

# 제출할 놈
y_submit = model.predict(test_csv)
# print(y_submit)                                           # nan이 출력되는 걸로 봐선 test_csv에도 뭔가 결측치가 있었다는 뜻임.
# print(y_submit.shape)  

# .to_csv를 사용해서
# submission_0105.csv를 완성하시오!!!

# print(submission)
submission['count'] = y_submit                              # 대여수를 예측한 카운트에 submission에 넣어준다.
# print(submission)

submission.to_csv(path + 'submission_01091627.csv')         # 날짜.시간 01061152 = 1월 6일 11시 52분

'''
CPU
:

GPU
:

'''