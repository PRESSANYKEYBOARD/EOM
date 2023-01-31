import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator                     # 이미지를 전처리한다.

train_dir = 'd:/_data/train'
test_dir = 'd:/_data/test1'

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

xy_test = test_datagen.flow_from_directory(                                     
    test_dir,
    target_size=(150, 150),
    batch_size=100,                                                                       
    class_mode='binary',                                                              
    color_mode='rgb',                                                              
    shuffle=True,                                                                       
    # Found 12500 images belonging to 1 classes.                                                                               
)                                                                                        

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000022DF3623E80>

print(xy_train[0][0].shape)                                                         # (100, 150, 150, 3)
print(xy_train[0][1].shape)                                                         # (100,)

print(xy_test[0][0].shape)                                                          # (100, 150, 150, 3)
print(xy_test[0][1].shape)                                                          # (100,)

np.save('d:/_data/brain_x_train.npy', arr=xy_train[0][0])
np.save('d:/_data/brain_y_train.npy', arr=xy_train[0][1])
# np.save('./_data/brain/brain_xy_train.npy', arr=xy_train[0])                                        # 분리해서 빼야한다.

np.save('d:/_data/brain_x_test.npy', arr=xy_test[0][0])
np.save('d:/_data/brain_y_test.npy', arr=xy_test[0][1])