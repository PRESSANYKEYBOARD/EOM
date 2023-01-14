# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1 .데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)                                             # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2
)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(5, input_shape=(8, )))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
start = time.time()
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=50, batch_size=1,
          validation_split=0.2,
          verbose=1)                                                # verbose= 진행표시줄 on/off
                                                                    # default=1
                                                                    # hist = history
                                                                    # model.fit 값을 받아서 hist라는 변수로 리턴한다.

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

print("==============================")                                               
print(hist)                                                         # <keras.callbacks.History object at 0x000001BF527C0AF0>
print("==============================")
print(hist.history)                                                 # hist 공간 안에 history 라는 제공된 변수가 있다.
                                                                    # loss와 val_loss 값이 딕셔너리 형태로 들어가 있다.
                                                                    # 파이썬의 데이터 형태는 list / key / value
                                                                    # dictionary(딕셔너리): 형태와 {} 형태로 묶여져 있는데, value 값이 list 형태로 묶여져 있다.
                                                                    # 딕셔너리는 key와 value 형태 / 홍길동: {국영수 점수} / 심청이 :{국영수 점수}
                                                                    # 두 개 이상=list 형태
                                                                    
                                                                    # model.fit은 훈련의 결과값을 반환하고 그걸 hist(history)라 하자.
                                                                    # print(hist.history)하면 loss, val_loss, metrics 등을 dictionary 형태로 보여준다.
                                                                    # dictionary : {'분류이름(key)' : [ , , , , ...], 'val_loss' : [ , , , , ...] (value)...} : key, value 형태이다.
                                                                    
print("==============================")
print(hist.history['loss'])                                         # loss 값 출력
print("==============================")
print(hist.history['val_loss'])                                     # val_loss 값 출력

import matplotlib.pyplot as plt                                     # (epochs, loss)의 산점도 및 그래프를 작성할 수 있음

plt.figure(figsize=(9,6))                                           # 그림에 대한 판 사이즈
plt.plot(hist.history['loss'], c='red', 
         marker='.', label='loss')                                  # list 형태에서 x 명시는 굳이 안 해도 상관 없다. y만 넣어주면 된다.
                                                                    # c= 색깔 지정
                                                                    # marker= 선의 모양
                                                                    # label= 선의 이름
                                                
plt.plot(hist.history['val_loss'], c='blue', 
         marker='.', label='val_loss')              
               
plt.grid()                                                          # 격자 넣기
plt.xlabel('epochs')                                                # x축을 epochs 지정
plt.ylabel('loss')                                                  # y축을 loss 지정
plt.title('california loss')                                            # 제목을 지정
plt.legend()                                                        # 라벨이 나오게 된다 / 그래프가 없는 지점에 나온다.
# plt.legend(loc='upper left')                                      # loc= 라벨을 어디에 나오게? / upper right
plt.show()                                                          # 그림 보여주기


"""
<loss, val_loss를 통해 훈련이 잘 되는지 확인하기>
loss값을 참고하되 val_loss가 기준이 된다.
val_loss가 들쭉날쭉하므로 훈련이 잘 안되는 중이다.
val_loss가 최소인 지점이 최적의 weight 점이다.

<램의 용량과 연산량>
model.add(Dense(40000))
model.add(Dense(30000))
: 4만 곱하기 3만으로 연산량 약 12억이므로 중간에 메모리 부족하다고 오류 뜸.

model.add(Dense(40000))
model.add(Dense(3))
model.add(Dense(30000))
: 4만 곱하기 3, 3 곱하기 3만 이므로 최대치 약 12만으로 메모리 안 부족함.

.
.
.

<matplotlib 한글 깨짐>
윈도우 PC에서는 폰트가 C:\Windows\Fonts에 위치한다.\
여기서 쓰고자 하는 폰트의 속성에 들어가 폰트의 영문이름을 확인한다.
ex) 맑은 고딕 보통은 malgun.ttf 이다.

1번째 방법
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

2번째 방법
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

"""