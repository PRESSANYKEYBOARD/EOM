import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 전처리한다.

train_dir = 'd:/_data/train'
test_dir = 'd:/_data/test1'

train_datagen = ImageDataGenerator(
    rescale=1./255,                                                                   # 이미지를 minmax 하겠다
)

test_datagen = ImageDataGenerator(                                                      # rescale만 한다.
    rescale=1./255,                                                                      # 정확한 평가를 하기 위해서 증폭되지 않은 데이터를 가지고 평가한다.
                                                                                        # 증폭할 필요가 없다...                                                                                                         
)

xy_train = train_datagen.flow_from_directory(                                            # 폴더 내의 이미지 데이터를 가져오겠다. 
    train_dir,
    target_size=(150, 150),
    batch_size=100,                                                                      # x = (160장, 이미지크기 150, 150, 흑백 1)
    class_mode='binary',                                                                # y = (160, )
    color_mode='rgb',                                                              # np.unique = 1:80
    shuffle=True,                                                                       # 데이터가 0이 80장, 1이 80장
    # Found 25000 images belonging to 2 classes.                                                                                  
                                                                                        # (200, 200으로 증폭) / (100, 100으로 하면 압축)
                                                                                        # 파이토치에서는 배치를 미리 분리해둔다.
)                                            

# np.save('d:/_data/brain_x_train.npy', arr=xy_train[0][0])
# np.save('d:/_data/brain_y_train.npy', arr=xy_train[0][1])
# # np.save('./_data/brain/brain_xy_train.npy', arr=xy_train[0])                                        # 분리해서 빼야한다.

# np.save('d:/_data/brain_x_test.npy', arr=xy_test[0][0])
# np.save('d:/_data/brain_y_test.npy', arr=xy_test[0][1])

x_train = np.load('d:/_data/brain_x_train.npy')
y_train = np.load('d:/_data/brain_y_train.npy')
x_test = np.load('d:/_data/brain_x_test.npy')
y_test = np.load('d:/_data/brain_y_test.npy')

print(x_train.shape,y_train.shape)                                  # (10, 150, 150, 3) (10,)
print(x_test.shape,y_test.shape)                                    # (10, 150, 150, 3) (10,)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
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
#     x_batch, y_batch = xy_batch
#     print('x_batch shape:', x_batch.shape)
#     break

# fig, axes = plt.subplots(1, 100, figsize=(20, 8))
# for idx, img_data in enumerate(x_batch[:100]):
#     axes[idx].imshow(img_data)

# plt.tight_layout()
# plt.show()
