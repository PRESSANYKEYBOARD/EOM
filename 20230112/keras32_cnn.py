from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten                      # 이미지쪽은 Conv2D로 한다.

model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(2, 2),                                # Conv2D에는 필터가 10개 있고, 커널 사이즈를 2x2로 하겠다. / kernel_size= 이미지를 조각내는 사이즈
                input_shape=(5, 5, 1)))                                         # 5x5 짜리 필터 하나가 들어가는데 10개로 늘리겠다.
                                                                                # 하이퍼 파라미터 튜닝은 다 들어간다. 단지 다른점은 이미지를 어떻게 받아오고 Dense 형식의 데이터로 변경되느냐???
                                                                                
model.add(Conv2D(5, kernel_size=(2, 2)))                                
model.add(Flatten())                                                            # flatten은 numpy에서 제공하는 다차원 배열 공간을 1차원으로 평탄화해주는 함수이다. / 쫙 펼쳐준다.
model.add(Dense(10))
model.add(Dense(1))

model.summary()
                                                                                             
