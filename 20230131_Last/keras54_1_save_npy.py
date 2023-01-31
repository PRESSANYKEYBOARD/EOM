# 53_5 복붙

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 전처리한다.

train_datagen = ImageDataGenerator(
    rescale=1./255,                                                                   # 이미지를 minmax 하겠다.
    # horizontal_flip=True,                                                               # 수평반전
    # vertical_flip=True,                                                                  # 수직반전
    # width_shift_range=0.1,                                                              # 10% 만큼 이동
    # height_shift_range=0.1,
    # rotation_range=0.5,                                                                 # 이미지 회전
    # zoom_range=1.2,                                                                     # 원래 그림의 20% 확대 
    # shear_range=0.7,                                                                    # 
    # fill_mode='nearest'
)

                                                                                        # 원본 데이터 그대로 리스케일만 진행하고 나머지는 모두 주석처리한다.

test_datagen = ImageDataGenerator(                                                      # rescale만 한다.
    rescale=1./255                                                                      # 정확한 평가를 하기 위해서 증폭되지 않은 데이터를 가지고 평가한다.
                                                                                        # 증폭할 필요가 없다...                                                                                                         
)

xy_train = train_datagen.flow_from_directory(                                           # 폴더 내의 이미지 데이터를 가져오겠다. 
    './_data/brain/train',
    target_size=(200, 200),
    batch_size=10000,                                                                   # x = (160장, 이미지크기 150, 150, 흑백 1)
    class_mode='binary',                                                                # y = (160, ),    
    # class_mode='categorical',                                                         # 0과 1로 되어있으니 바이너리로.                                                            # y = (160, )
    color_mode='grayscale',                                                             # np.unique = 1:80
    shuffle=True,                                                                       # 데이터가 0이 80장, 1이 80장
    # Found 160 images belonging to 2 classes.                                                                                    
                                                                                        # (200, 200으로 증폭) / (100, 100으로 하면 압축)
                                                                                        # 파이토치에서는 배치를 미리 분리해둔다.
)                                            

xy_test = test_datagen.flow_from_directory(                                     
    './_data/brain/test',
    target_size=(200, 200),
    batch_size=10000,
    class_mode='binary',                                                                                                                                    
    # class_mode='categorical',                                                              
    color_mode='grayscale',                                                              
    shuffle=True,                                                                       
    # Found 120 images belonging to 2 classes.                                                                                   
)                                                                                        

print(xy_train)
# # <keras.preprocessing.image.DirectoryIterator object at 0x00000217DF247B50>

# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets)

np.save('C:/study/_data/brain/brain_x_train.npy', arr=xy_train[0][0])
np.save('C:/study/_data/brain/brain_y_train.npy', arr=xy_train[0][1])
# np.save('./_data/brain/brain_xy_train.npy', arr=xy_train[0])                                        # 분리해서 빼야한다.

np.save('C:/study/_data/brain/brain_x_test.npy', arr=xy_test[0][0])
np.save('C:/study/_data/brain/brain_y_test.npy', arr=xy_test[0][1])

# print(xy_train[0])
# print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[0][0].shape)                                    # (10, 200, 200, 1)
print(xy_train[0][1].shape)                                    # (10, 2)   

#                                                                 # xy_train의 배치사이즈를 확 늘리면 (160, 200, 200, 1)로 출력된다.
                                                                
# print(type(xy_train))                                           # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))                                        # <class 'tuple'> = 리스트와 똑같다.
#                                                                 # 튜플을 한번 생성하면 바꾸질 못한다.
                                                                
# print(type(xy_train[0][0]))                                     # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))                                     # <class 'numpy.ndarray'>
