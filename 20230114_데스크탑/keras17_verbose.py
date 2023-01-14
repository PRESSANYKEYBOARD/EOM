# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1 .데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)                                             # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2
)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(5, input_shape=(13, )))
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
          verbose=1)                                                # verbose= 진행표시줄 on/off
                                                                    # default=1
                                                                    # True : 1, False : 0, 프로그래스바 제거(진행바 사라짐) : 2, 에포(반복치)만 보여줌 : 3 ~
                                                                    # 말수가 많음 실행할 때, 코드 보여주는 게 딜레이가 생긴다.
                                                                    # 자원낭비
                                                                    
                                                                    # 함수 수행 시 발생하는 상세한 정보들을 표준 출력으로 자세히 내보낼 것인가를 나타냄
                                                                    # 0: 미출력, 1(Default): 자세히, 2: 함축적 정보 출력 3. 2보다 더 함축적 정보 출력
                                                                    
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

print("걸린시간:", end - start)                                      # verbose = 학습 중 출력되는 문구를 설정합니다.
                                                                        # - 0 : 아무 것도 출력하지 않습니다.
                                                                        # - 1 : 훈련의 진행도를 보여주는 진행 막대를 보여줍니다.
                                                                        # - 2 : 미니 배치마다 손실 정보를 출력합니다.

                                                                    # verbose=1 / loss: 47.06433868408203, 걸린시간: 13.293114423751831
                                                                    # verbose=0 / loss: 53.02677536010742, 걸린시간: 10.317590236663818
                                                                    # verbose=2 / loss: 57.73585891723633, 걸린시간: 10.568190813064575
                                                                    # verbose=3 / loss: 58.02903366088867, 걸린시간: 10.319077253341675