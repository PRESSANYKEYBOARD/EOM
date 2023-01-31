# 개 사진 고양이 사진 한 개를 인터넷에서 잘라내서 뭔지 맞추기

from typing import Sequence
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping

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

xy_train = train_datagen.flow_from_directory(                                           # 폴더 내의 이미지 데이터를 가져오겠다. 
    train_dir,
    target_size=(50, 50),
    batch_size=100,                                                                     # x = (160장, 이미지크기 150, 150, 흑백 1)
    class_mode='binary',                                                                # y = (160, )
    color_mode='rgb',                                                                   # np.unique = 1:80
    shuffle=True,                                                                       # 데이터가 0이 80장, 1이 80장
    # Found 25000 images belonging to 2 classes.                                                                                  
                                                                                        # (200, 200으로 증폭) / (100, 100으로 하면 압축)
                                                                                        # 파이토치에서는 배치를 미리 분리해둔다.
)                                            

xy_test = test_datagen.flow_from_directory(                                     
    test_dir,
    target_size=(50, 50),
    batch_size=100,                                                                       
    class_mode='binary',                                                              
    color_mode='rgb',                                                              
    shuffle=True,                                                                       
    # Found 12500 images belonging to 1 classes.                                                                               
)                                                                                        

'''
flow_from_directory 메소드를 사용하면 폴더구조를 그대로 가져와서 ImageDataGenerator 객체의 실제 데이터를 채워준다. 
이 데이터를 불러올 때 앞서 정의한 파라미터로 전처리를 한다.
'''

print(type(xy_train)) 
# <class 'keras.preprocessing.image.DirectoryIterator'>
print(xy_train[0][0].shape)                                                             # (100, 150, 150, 3)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(50, 50, 3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 
model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
hist = model.fit(xy_train, epochs=10, steps_per_epoch=20, validation_data=xy_train, validation_steps=40, callbacks=[es]) 

#4. 예측
sample_directory = 'D:/_data/'
sample_image = sample_directory + "test3.jpg"

print("-- Evaluate --")
scores = model.evaluate_generator(xy_test, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
image_ = image.load_img(str(sample_image), target_size=(50, 50))
x = image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)

print(classes)

xy_test.reset()
print(xy_test.class_indices)
# {'cats': 0, 'dogs': 1}

if(classes[0][0]<=0.5):
    cat = 100 - classes[0][0]*100
    print(f"당신은 {round(cat,2)} % 확률로 고양이 입니다.")
elif(classes[0][0]>=0.5):
    dog = classes[0][0]*100
    print(f"당신은 {round(dog,2)} % 확률로 개 입니다.")
else:
    print("ERROR")