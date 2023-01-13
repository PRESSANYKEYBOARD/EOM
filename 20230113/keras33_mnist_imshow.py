import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)                                 # (60000, 28, 28) (60000,)
                                                                    # x에는 60000장의 흑백 데이터
                                                                    # 데이터 내용이나 순서의 영향을 받지 않는다. reshape
print(x_test.shape, y_test.shape)                                   # (10000, 28, 28) (10000,)

print(x_train[1000])
print(y_train[1000])                                                # 5

import matplotlib.pyplot as plt
plt.imshow(x_train[1000], 'gray')
plt.show()                                                          