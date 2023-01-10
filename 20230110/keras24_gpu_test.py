import tensorflow as tf
print(tf.__version__)               # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')          # 테스트 버전
                                                                    # 텐서플로의 물리적인 장치 GPU를 변수안에 값을 넣어주겠다.
print(gpus)

if(gpus):
    print("쥐피유 돈다")
else:
    print("쥐피유 안돈다")