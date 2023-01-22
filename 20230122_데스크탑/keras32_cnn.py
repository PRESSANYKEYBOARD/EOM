# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten                      # 이미지쪽은 Conv2D로 한다.

model = Sequential()
                                                                                # 인풋은 (60000, 5, 5, 1)
                                                                                # 가로 5, 세로, 5 컬러 이미지 1개인 60000개가 있다. 

model.add(Conv2D(filters=10, kernel_size=(2, 2),                                # Conv2D에는 필터가 10개 있고, 커널 사이즈를 2x2로 하겠다. / kernel_size= 이미지를 조각내는 사이즈
                input_shape=(5, 5, 1)))                                         # 5x5 짜리 필터 하나가 들어가는데 10개로 늘리겠다.
                                                                                # 하이퍼 파라미터 튜닝은 다 들어간다. 단지 다른점은 이미지를 어떻게 받아오고 Dense 형식의 데이터로 변경되느냐???
                                                                                # (N, 4, 4, 10)
                                                                                # (batch_size, rows, columns, channels 또는 필터)
                                                                                # batch_size(훈련의 갯수) 단위로 연산한다.
                                                                                
                                                                                # 이미지 픽셀 수(5*5) 1개(흑백) 3개(컬러RPG)
                                                                                # kernel_size = 합을 연산할 이미지 블럭의 사이즈(2*2)
                                                                                # 1 layer 연산 후 5*5 -> 4*4 -> ... 연산량이 점점 줄어감
                                                                                # filter = 10: 5*5 흑백 이미지 1장을 10장으로 늘림
                                                                                # hyper-parameter tuning: filters, kernel_size, activation 등

                                                                                
model.add(Conv2D(5, kernel_size=(2, 2)))                                        # (N, 3, 3, 5)

model.add(Flatten())                                                            # flatten은 numpy에서 제공하는 다차원 배열 공간을 1차원으로 평탄화해주는 함수이다. / 쫙 펼쳐준다.
                                                                                # (N, 45)
                                                                                
model.add(Dense(units=10))                                                      # (N, 10)
                                                                                # 인풋은 (batch_size, input=dim)
                                                                                # Nan = 고정적으로 제공되는 Data의 양
                                                                                # model.add(tf.keras.layers.Dense(32))
                                                                                # model.output_shape (None, 32) -> None = data의 개수

model.add(Dense(4, activation='relu'))                                          # (N, 1)

model.summary()

                                                                                             
"""
 Layer (type)                Output Shape              Param #
 conv2d (Conv2D)             (None, 4, 4, 10)          50                       # 5 x 5 x 커널 사이즈 2 = 50
 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205                      # 160 + 45
 flatten (Flatten)           (None, 45)                0                        # 0
 dense (Dense)               (None, 10)                460
 dense_1 (Dense)             (None, 1)                 11                       # 
 
input_shape : 샘플 수를 제외한 입력 형태를 정의 합니다. 모델에서 첫 레이어일 때만 정의하면 됩니다.
(행, 열, 채널 수)로 정의합니다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다.
input_shape=(None, 가로픽셀, 세로픽셀, feature) 1흑백(깊이1) 3컬러 None개 사진을 훈련.
Output Shape은 (행 수, 가로 픽셀 수, 세로 픽셀 수, 채널 수) 라고 생각하면 쉽다.
여기서 행(row) 수는 None으로 표시되어 있는데, 이는 특정한 숫자로 지정되지 않았음을 의미한다. 배치 사이즈처럼 32, 64 등 다양한 숫자가 올 수도 있기 때문이다.
채널 수는 각 Conv2D에서 지정한 filters의 값에 맞춰 나온 것을 볼 수 있다.
          
"""