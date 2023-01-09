from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()                                    # 소문자는 함수, 대문자는 클래스
x = datasets.data
y = datasets.target
print(x.shape, y.shape)                                     # (506, 13) (506,) 
                                                            # 행 무시, 열 우선 / input_dim = 13
                                                            
x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=333, test_size=0.2 
)                                                           # train은 대략 404개 정도

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))                           # input_dim은 행과 열만 되어있는 곳만 쓸 수 있다.
model.add(Dense(5, input_shape=(13, )))                     # 스칼라가 13개로 볼 수 있기 때문에, (13, ) 라고 해석할 수 있다.
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
model.fit(x_train, y_train, epochs=50, batch_size=1,
          validation_split=0.2, 
          verbose=3)                                        # verbose= 진행표시줄 on/off
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

print("걸린시간:", end - start)                             
                                                            # verbose=1 걸린시간: 15.09028697013855         # 진행바가 보임 / 모든게 다 나옴
                                                            # verbose=0 걸린시간: 11.766844987869263         # 진행바 그거 뭐임? / 훈련 과정이 안 보임
                                                            # verbose=2 걸린시간: 15.09028697013855          # loss, metrics값만 나오고 진행바는 안 보임
                                                            # verbose=3 걸린시간: 12.230494022369385         # epoch만 진행됨
                                                            # verbose=4 걸린시간: 14.60512399673462          # epoch만 진행됨