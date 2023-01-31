# 53_1 복붙

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 전처리한다.
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
augment_size = 100                                                                      # 100장으로 증폭

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

x_data = train_datagen.flow(                                                          # 폴더 내의 이미지 데이터를 가져오겠다. 
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),            # x       # (100, 784)
    np.zeros(augment_size),                                                             # y       
    batch_size=augment_size,
    shuffle=True,                          
)                                                                            

print(x_data[0])
print(x_data[0][0].shape)
print(x_data[0][1].shape)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
plt.show()
