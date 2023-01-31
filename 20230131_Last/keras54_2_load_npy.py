# 54_2 복붙

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 전처리한다.

# np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])
# np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])
# # np.save('./_data/brain/brain_xy_train.npy', arr=xy_train[0])                                        # 분리해서 빼야한다.

# np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
# np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])

x_train = np.load('C:/study/_data/brain/brain_x_train.npy')
y_train = np.load('C:/study/_data/brain/brain_y_train.npy')
x_test = np.load('C:/study/_data/brain/brain_x_test.npy')
y_test = np.load('C:/study/_data/brain/brain_y_test.npy')

print(x_train.shape, y_train.shape)                                 # (160, 200, 200, 1) (160,)
print(x_test.shape, y_test.shape)                                   # (120,) (120,)
# print(x_train[100])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(200, 200, 1), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

# hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=100,
#                     validation_data=xy_test,
#                     validation_steps=4, )                                                   # epochs 당 배치 몇번?

hist = model.fit(x_train, y_train,
                 batch_size=16,
                #  steps_per_epoch=16, 
                 epochs=100,
                validation_data=([x_test, y_test])
                # validation_steps=4, 
)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print("loss: ", loss[-1])
print("val_loss: ", val_loss[-1])
print("accuracy: ", acc[-1])
print("val_acc: ", val_acc[-1])

# # 그림그려라!!! matplybit 완성시켜라

# import matplotlib.pyplot as plt        

# for xy_batch in x_train:
#     x, y = xy_batch
#     print('x_data의 shape : {}'.format(x.shape))
#     break                                                                     # for를 탈출

# fig, axes = plt.subplots(1, 10, figsize=(20, 8))
# for idx, img_data in enumerate(x[:10]):                                       # enumerate: 리스트가 있는 경우 순서와 리스트의 값을 전달하는 기능을 가집니다.
#     axes[idx].imshow(img_data)

# plt.tight_layout()
# plt.show()

"""
loss:  0.0052272239699959755
val_loss:  2.070427894592285    
accuracy:  1.0
val_acc:  0.625
xy_data의 shape : (160, 100, 100, 1)

loss:  0.46013355255126953
val_loss:  3.819855213165283
accuracy:  0.9937499761581421
val_acc:  0.9666666388511658

"""                                                                          
