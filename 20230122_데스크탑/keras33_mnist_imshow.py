# 데스크탑에서 세팅하고 다시 복습용으로 생성했습니다. 블루스크린 나쁜 놈 ㅠㅠㅠ

# CPU: intel Core i7-8700 3.20Ghz (6C12T)
# GPU: Nvidia Geforce RTX2070 SUPER
# RAM: Samsung DDR4-3200 16GB x2 = 32GB
# SSD: ADATA 8200XP 1TB
# HDD: Western Digital 1TB 7200rpm

import numpy as np
from tensorflow.keras.datasets import mnist                                                 # mnist: 고등학생과 미국 인구조사국 직원들이 손으로 쓴 70,000개의 작은 숫자 이미지들의 집합

#

(x_train, y_train), (x_test, y_test) = mnist.load_data()                                    # 교육용 자료, 이미 train/test 분류

print(x_train.shape, y_train.shape)                                                         # (60000, 28, 28) (60000,) reshape (훈련)
print(x_test.shape, y_test.shape)                                                           # (10000, 28, 28) (10000,) (테스트)

print(x_train[1000])
print(y_train[1000])

import matplotlib.pyplot as plt
plt.imshow(x_train[0], 'gray')
plt.show()