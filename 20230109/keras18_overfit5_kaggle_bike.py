from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split

#1. 데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # 데이터에서 제외할 인덱스 칼럼이 0열에 있음. 열은 데이터가 아닌 인덱스라고 알려줘서 데이터로 들어가는 것을 방지.
                                                                # train_csv = pd.read_csv('./_data/ddarung/train.csv,' index_col=0)
test_csv = pd.read_csv(path +'test.csv', index_col=0)           # index_col 지정하지 않으면, 인덱스가 자동으로 생성된다.
submission = pd.read_csv(path + 'samplesubmission.csv', index_col=0)

print(train_csv)                                                # [10886 rows x 11 columns]
print(train_csv.shape)                                          # (10886, 11)
print(submission.shape)                                         # (6493, 1)

print(train_csv.columns)
                                                                # (['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                                                                    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
                                                                    #   dtype='object')
print(train_csv.info()) 
print(test_csv.info()) 
print(train_csv.describe())                                     # [8 rows x 11 columns]

#### 결측치 처리 1. 제거 ####
print(train_csv.isnull().sum())                                 # 중단점 클릭하고 F5 하면 중단점 직전까지 실행이 된다.

'''
결측치 데이터
season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
casual        0
registered    0
count         0

'''

train_csv = train_csv.dropna()
print(train_csv.isnull().sum())                                     # 결측값 확인
print(train_csv.shape)  # (10886, 11)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)       # 작업할 때 axis가 1이면 행, 0이면 열
                                                                    # train.csv와 test.csv를 비교했을 때 casual, registered, count가 있고 없고의 차이임. 이걸 제거해주는 작업임.

print(x)                                                            # [10886 rows x 10 columns]
y = train_csv['count']
print(y)                                                            # (10886, 11)
print(y.shape)                                                      # (10886,)
                                                           
x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, test_size=0.2 
)                                                               # train은 대략 8,708개 정도

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=11))                               # input_dim은 행과 열만 되어있는 곳만 쓸 수 있다.
model.add(Dense(5, input_shape=(11, )))                         # 스칼라가 11개로 볼 수 있기 때문에, (11, ) 라고 해석할 수 있다.
                                                                # (100, 10, 5) 라는 데이터가 들어가면, 행 무시 열 우선이기 때문에 (10, 5)로 들어감.
                                                                # 앞으로는 이렇게 쓰자!!!
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
start = time.time()
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_split=0.2, 
          verbose=1)                                            # verbose= 진행표시줄 on/off
                                                                # default=1
                                                                # hist = history
                                                                # model.fit 값을 받아서 hist라는 변수로 리턴한다.
                                                            
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

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
         marker='.', label='loss')                  # list 형태에서 x 명시는 굳이 안 해도 상관 없다. y만 넣어주면 된다.
                                                    # c= 색깔 지정
                                                    # marker= 선의 모양
                                                    # label= 선의 이름
                                                
plt.plot(hist.history['val_loss'], c='blue', 
         marker='.', label='val_loss')              
               
plt.grid()                                                  # 격자 넣기
plt.xlabel('epochs')                                        # x축을 epochs 지정
plt.ylabel('loss')                                          # y축을 loss 지정
plt.title('boston loss')                                    # 제목을 지정
plt.legend()                                                # 라벨이 나오게 된다 / 그래프가 없는 지점에 나온다.
# plt.legend(loc='upper left')                              # loc= 라벨을 어디에 나오게? / upper right
plt.show()                                                  # 그림 보여주기


'''
결과는 val_loss 값이 들쭉날쭉하다.
loss값은 참고로 하되, val_loss값을 기준으로 해라.

'''

