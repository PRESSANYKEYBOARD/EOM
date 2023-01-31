# 57_1 복붙

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 전처리한다.
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
augment_size = 40000                                                                    # 40000장으로 증폭
randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)
print(len(randidx))                                                                     # 60000개 중 40000개의 이미지 추출

x_augument = x_train[randidx].copy()                                                    # 데이터의 원본을 건들지 않고 복사본으로 작업
y_augument = y_train[randidx].copy()
print(x_augument.shape, y_augument.shape)                                               # (40000, 28, 28) (40000,)

x_augument = x_augument.reshape(40000, 28, 28, 1)

train_datagen = ImageDataGenerator(
    rescale=1./255,                                                                   # 이미지를 minmax 하겠다.
    horizontal_flip=True,                                                               # 수평반전
    # vertical_flip=True,                                                                  # 수직반전
    width_shift_range=0.1,                                                              # 10% 만큼 이동
    height_shift_range=0.1,
    rotation_range=0.5,                                                                 # 이미지 회전
    # zoom_range=1.2,                                                                     # 원래 그림의 20% 확대 
    shear_range=0.7,                                                                    # 
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(                                                      # rescale만 한다.
    rescale=1./255                                                                      # 정확한 평가를 하기 위해서 증폭되지 않은 데이터를 가지고 평가한다.
                                                                                        # 증폭할 필요가 없다...                                                                                                         
)

x_augumented = train_datagen.flow(                                                            # 폴더 내의 이미지 데이터를 가져오겠다. 
    x_augument,                                                                               # x       # (100, 784)
    y_augument,
    batch_size=augment_size,
    shuffle=True,                          
)                                                                            

print(x_augumented[0][0].shape)
print(x_augumented[0][1].shape)

x_train = x_train.reshape(60000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented[0][0]))
y_train = np.concatenate((y_train, x_augumented[0][1]))

print(x_train.shape, y_train.shape)                                                          # (100000, 28, 28, 1) (100000,)