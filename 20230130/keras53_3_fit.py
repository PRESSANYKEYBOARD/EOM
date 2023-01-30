#53_2 복붙

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 전처리한다.

train_datagen = ImageDataGenerator(
    rescale=1./255,                                                                   # 이미지를 minmax 하겠다.
    horizontal_flip=True,                                                               # 수평반전
    vertical_flip=True,                                                                  # 수직반전
    width_shift_range=0.1,                                                              # 10% 만큼 이동
    height_shift_range=0.1,
    rotation_range=0.5,                                                                 # 이미지 회전
    zoom_range=1.2,                                                                     # 원래 그림의 20% 확대 
    shear_range=0.7,                                                                    # 
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(                                                      # rescale만 한다.
    rescale=1./255                                                                      # 정확한 평가를 하기 위해서 증폭되지 않은 데이터를 가지고 평가한다.
                                                                                        # 증폭할 필요가 없다...                                                                                                         
)

xy_train = train_datagen.flow_from_directory(                                            # 폴더 내의 이미지 데이터를 가져오겠다. 
    './_data/brain/train',
    target_size=(100, 100),
    batch_size=1000,                                                                      # x = (160장, 이미지크기 150, 150, 흑백 1)
    class_mode='binary',                                                                # y = (160, )
    color_mode='grayscale',                                                              # np.unique = 1:80
    shuffle=True,                                                                       # 데이터가 0이 80장, 1이 80장
    # Found 160 images belonging to 2 classes.                                                                                    
                                                                                        # (200, 200으로 증폭) / (100, 100으로 하면 압축)
                                                                                        # 파이토치에서는 배치를 미리 분리해둔다.
)                                            

xy_test = test_datagen.flow_from_directory(                                     
    './_data/brain/test',
    target_size=(100, 100),
    batch_size=1000,                                                                      
    class_mode='binary',                                                              
    color_mode='grayscale',                                                              
    shuffle=True,                                                                   # 0과 1이 적당히 섞여져야..                                                                       
    # Found 120 images belonging to 2 classes.                                                                                   
)                                                                                        

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(100, 100, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

# hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=100,
#                     validation_data=xy_test,
#                     validation_steps=4, )                                                   # epochs 당 배치 몇번?

hist = model.fit(xy_train[0][0], xy_train[0][1],
                 batch_size=16,
                #  steps_per_epoch=16, 
                 epochs=100,
                validation_data=(xy_test[0][0], xy_test[0][1])
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

# 그림그려라!!! matplybit 완성시켜라

import matplotlib.pyplot as plt        

for xy_batch in xy_train:
    x, y = xy_batch
    print('xy_data의 shape : {}'.format(x.shape))
    break                                                                     # for를 탈출

fig, axes = plt.subplots(1, 10, figsize=(20, 8))
for idx, img_data in enumerate(x[:10]):                                       # enumerate: 리스트가 있는 경우 순서와 리스트의 값을 전달하는 기능을 가집니다.
    axes[idx].imshow(img_data)

plt.tight_layout()
plt.show()                                                                          