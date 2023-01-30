import numpy as np
import pandas as pd

#1. 데이터

# 1-1. csv 파일 불러오기
path = 'C:/study/_data/Jusik_Mang/'                            # 절대 경로
samsung = pd.read_csv(path + '삼성전자 주가.csv', thousands=',', encoding='euc-kr') 
amore = pd.read_csv(path +'아모레퍼시픽 주가.csv', thousands=',', encoding='euc-kr')

# 1-2. 데이터 확인
# print(samsung.head())
# print("=============================================================================================")
# print(amore.head())

#            일자     시가     고가     저가     종가 전일비  Unnamed: 6   등락률         거래량     금액(백만)   신용비       개인       기관   외
# 인(수량)      외국계     프로그램    외인비
# 0  2023/01/27  64400  65000  63900  64600   ▲         700  1.10  18154371.0  1173646.0  0.00 -6139035   447761        0  5317461  1775555  50.34 
# 1  2023/01/26  63800  63900  63300  63900   ▲         500  0.79  13278277.0   846409.0  0.09 -3441185   -56761  3474033  3332083  2663007  50.34 
# 2  2023/01/25  63500  63700  63000  63400   ▲        1600  2.59  16822710.0  1066201.0  0.09 -7609753   164892  6569212  6060813  3336405  50.28 
# 3  2023/01/20  62100  62300  61100  61800   ▲         300  0.49   9646327.0   595373.0  0.10 -1647820   127097  1225750   706062   759602  50.17 
# 4  2023/01/19  60500  61500  60400  61500   ▲        1100  1.82  12808490.0   781938.0  0.10 -3306867 -1422690  3273083  4612873  3046155  50.15 
# =============================================================================================
#            일자      시가      고가      저가      종가 전일비  Unnamed: 6   등락률       거래량   금액(백만)   신용비     개인     기관  외인(수
# 량)    외국계   프로그램    외인비
# 0  2023/01/27  143600  149500  142600  147400   ▲        4000  2.79  306655.0  44926.0  0.00 -96741  40188       0  25627  53432  25.38
# 1  2023/01/26  145700  146200  142600  143400   ▼       -2200 -1.51  306126.0  43996.0  0.26  64458 -39574    1461 -49158  18604  25.38
# 2  2023/01/25  148100  149000  145000  145600   ▼       -2900 -1.95  206830.0  30260.0  0.27  46065 -35321  -31501   1883 -32156  25.38
# 3  2023/01/20  147500  149000  144000  148500   ▲        1500  1.02  234240.0  34523.0  0.28 -12988  26060   23107  21314 -12384  25.44
# 4  2023/01/19  142500  147500  141000  147000   ▲        3500  2.44  275781.0  40172.0  0.28 -54990  24696   30319  51322    623  25.40

# 1-3. 데이터 오름차순 정렬
samsung = samsung.loc[::-1].reset_index(drop = True)
amore = amore.loc[::-1].reset_index(drop=True)

# print(samsung.head())
# print("=============================================================================================")
# print(amore.head())

#            일자       시가       고가       저가       종가 전일비  Unnamed: 6   등락률       거래량    금액(백만)  신용비     개인     기관  외
# 인(수량)    외국계   프로그램    외인비
# 0  2015/01/13  1314000  1340000  1300000  1339000   ▲       23000  1.75  245868.0  324625.0  0.0 -20135 -21377   -7144  24273  17972  51.64      
# 1  2015/01/14  1339000  1355000  1335000  1345000   ▲        6000  0.45  286645.0  385455.0  0.0 -48993   7616   -4171  43638  -8546  51.64      
# 2  2015/01/15  1345000  1349000  1329000  1334000   ▼      -11000 -0.82  282078.0  378298.0  0.0 -11679   7790  -71115 -74997  15869  51.59      
# 3  2015/01/16  1334000  1334000  1313000  1316000   ▼      -18000 -1.35  271370.0  359887.0  0.0  -2134 -14849  -57261 -70419  24603  51.55      
# 4  2015/01/19  1329000  1349000  1320000  1343000   ▲       27000  2.05  133459.0  179082.0  0.0 -22450  26066   -3283  -4932   6940  51.55      
# =============================================================================================
#            일자       시가       고가       저가       종가 전일비  Unnamed: 6   등락률      거래량   금액(백만)  신용비    개인    기관  외인(수
# 량)   외국계  프로그램    외인비
# 0  2014/01/20   989000  1020000   985000  1019000   ▲       35000  3.56  18920.0  19089.0  0.0 -5589    65    5795  1197  9464  34.46
# 1  2014/01/21  1019000  1054000  1019000  1040000   ▲       21000  2.06  18645.0  19380.0  0.0 -4627 -1167    5614   403  2940  34.55
# 2  2014/01/22  1040000  1050000  1035000  1039000   ▼       -1000 -0.10   9282.0   9649.0  0.0  -706  -806    1386 -1258  -852  34.58
# 3  2014/01/23  1042000  1042000  1018000  1037000   ▼       -2000 -0.19   6730.0   6958.0  0.0  -289   580    -360  -131 -1199  34.57
# 4  2014/01/24  1023000  1046000  1022000  1030000   ▼       -7000 -0.68  11295.0  11706.0  0.0  -482  -261    1909 -1409 -1764  34.60

# 1-4. string 형태의 일자 index를 datetime으로 변경
samsung['일자'] = pd.to_datetime(samsung['일자'])
amore['일자'] = pd.to_datetime(amore['일자'])

# 1-5. '일자'를 연,월,일로 나누기 위한 연,월, 일 컬럼 추가
samsung.insert(0,'연',samsung['일자'].dt.year)
samsung.insert(1,'월',samsung['일자'].dt.month)
samsung.insert(2,'일',samsung['일자'].dt.day)

amore.insert(0,'연',amore['일자'].dt.year)
amore.insert(1,'월',amore['일자'].dt.month)
amore.insert(2,'일',amore['일자'].dt.day)

# 1-6. 기존 '일자' 컬럼 제거
samsung.drop(columns='일자', axis=1, inplace = True, errors='ignore')
amore.drop(columns='일자', axis=1, inplace = True, errors='ignore')

# 12개월의 데이터를 기준으로 잡고 그 이전 데이터를 모두 날려버려!!!! 하하핳
# 실제로는 회사 재무부터 시작해서 환율, 코스피 지수, 러시아-우크라이나 전쟁 진행, 기타 정치(EX. 영국 브렉시트) 등에 따라 변수가 많다.

# 엔화로 예를 들자면 1월 18일 오전 11시 40분경, 일본은행의 대규모 금융완화 정책을 유지한다는 발표함에 따라, 
# 100엔당 963원에서 950원대로 추락했다.

# 1-7. 2022년 1월 이전 데이터 삭제
  #1). 2022년 이전 데이터 삭제
delete_samsung_past_years = samsung[samsung['연']<2022].index
samsung.drop(delete_samsung_past_years, inplace = True, errors='ignore')
delete_amore_past_years = amore[amore['연']<2022].index
amore.drop(delete_amore_past_years,inplace = True, errors='ignore')

#   #2). 2022년 1월 이전 데이터 삭제                                                    # 1월부터 시작되니 굳이 안해도 된다. 2월부터이면 모를까...
# delete_samsung_past_months = samsung[samsung['월']<1].index 
# samsung.drop(delete_samsung_past_months,inplace = True, errors='ignore')
# delete_amore_past_months = amore[amore['월']<1].index
# amore.drop(delete_amore_past_months, inplace = True, errors='ignore')

# 1-8. 주가 예측 시 필요해 보이지 않는 컬럼 삭제 및 사용할 컬럼 정의
# [필요 없는 칼럼 삭제] 전일비, Unnamed: 6, 등락률, 금액(백만), 신용비, 개인, 기관, 외인(수량), 외국계, 프로그램, 외인비
samsung.drop(columns=['전일비', 'Unnamed: 6', '등락률', '금액(백만)','신용비', '개인', '기관', '외인(수량)',  '외국계', '프로그램', '외인비'], axis=1, inplace=True, errors='ignore')
amore.drop( columns=['전일비', 'Unnamed: 6', '등락률', '금액(백만)','신용비', '개인', '기관', '외인(수량)',  '외국계', '프로그램', '외인비'], axis=1, inplace=True, errors='ignore')

# [사용할 컬럼 정의] 사용할 column을 정의(쓸 컬럼만 지정 )
samsung = samsung.loc[:, ['연','월','일','시가', '고가', '저가', '종가', '거래량']]
amore = amore.loc[:, ['연','월','일','시가', '고가', '저가', '종가', '거래량']]

# 1-9. 데이터 자르기
samsung = np.array(samsung)
amore = np.array(amore)

def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column 
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number :y_end_number, 3]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
  
x1, y1 = split_xy(samsung, 5, 1)
x2, y2 = split_xy(amore, 5, 1)

print(x1.shape, y1.shape)                                                       # (259, 5, 8) (259, 1)
print(x2.shape, y2.shape)                                                       # (259, 5, 8) (259, 1)
  
# x1_datasets = np.array([range(100), range(301, 401)]).transpose()               # transpose = (2, 100)에서 (100, 2)로 변환
# print(x1_datasets.shape)                                                        # (100, 2) 
#                                                                                 # 삼성전자 시가, 고가
                                                                                
# x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose()
# print(x2_datasets.shape)                                                        # (100, 3) 
#                                                                                 # 아모레 시가, 고가, 종가

# y = np.array(range(2001, 2101))                                                 # (100, ) 
#                                                                                 # 삼성전자의 하루 뒤 종가

# 1-10. 데이터 reshape                                                                                
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(       # 3개 이상도 분리가 가능하다.
    x1, x2, y1, y2, train_size=0.7, shuffle=False, random_state=1234
)

print(x1_train.shape, x2_train.shape, y1_train.shape, y2_train.shape)                           # (181, 5, 8) (181, 5, 8) (181, 1) (181, 1)
print(x1_test.shape, x2_test.shape, y1_test.shape, y2_test.shape)                               # (78, 5, 8) (78, 5, 8) (78, 1) (78, 1)

x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2])
x1_test  = x1_test.reshape(x1_test.shape[0],x1_test.shape[1]*x1_test.shape[2])

x2_train = x2_train.reshape(x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2])
x2_test  = x2_test.reshape(x2_test.shape[0],x2_test.shape[1] * x2_test.shape[2])

# 1-11. 데이터 전처리
from sklearn.preprocessing import StandardScaler
data1 = StandardScaler()
data1.fit(x1_train)
x1_train_scale = data1.transform(x1_train)
x1_test_scale = data1.transform(x1_test)

data2 = StandardScaler()
data2.fit(x2_train)
x2_train_scale = data1.transform(x2_train)
x2_test_scale = data1.transform(x2_test)

# 1-12. 데이터 전처리 후 reshape
x1_train_scale = np.reshape(x1_train_scale,(x1_train_scale.shape[0], 5 , 8))
x1_test_scale = np.reshape(x1_test_scale,(x1_test_scale.shape[0], 5 , 8))
x2_train_scale = np.reshape(x2_train_scale,(x2_train_scale.shape[0], 5 , 8))
x2_test_scale = np.reshape(x2_test_scale,(x2_test_scale.shape[0], 5 , 8))

print(x2_train_scale.shape)                                                                     # (181, 5, 8)
print(x2_test_scale.shape)                                                                      # (78, 5, 8)


#2. 모델구성
from tensorflow.keras.models import Model                                                       
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, BatchNormalization

# 우선 Normalization 또는 정규화에 대해서 짧게 알아보자면, ML(Machine learning)을 돌리는 과정에서 
# Local optimum 문제에 빠져 잘못된 결과 산출이 발생할 수 있다.
# 이러한 문제를 줄이고자 우리는 Normalization, 즉 정규화를 통해 Global optimum 지점을 찾고 
# 원하는 결과를 산출해낼 수 있도록 컴퓨터에게 도움을 주는 것이다.

#2-1 모델1.
input1 = Input(shape=(5, 8))                                                                     # 행 무시 열 우선이니 (5, 8)
dense1 = LSTM(64, return_sequences=True, activation='linear', name='ds11')(input1)
dense2 = LSTM(64, activation='relu', name='ds12')(dense1)
dense3 = Dense(512, activation='relu', name='ds13')(dense2)
batch11 = BatchNormalization(name='bt11')(dense3)
dense4 = Dropout(0.3, name='ds14')(batch11)
dense5 = Dense(128, activation='relu', name='ds15')(dense4)
batch12 = BatchNormalization(name='bt12')(dense5)
dense6 = Dropout(0.3, name='ds16')(batch12)
dense7 = Dense(64, activation='relu', name='ds17')(dense6)
dense8 = Dropout(0.3, name='ds18')(dense7)
output1 = Dense(16, activation='softmax', name='ds19')(dense8)

#2-2 모델2.
input2 = Input(shape=(5, 8))
dense21 = LSTM(64, return_sequences=True, activation='linear', name='ds21')(input2)
dense22 = LSTM(64, activation='relu', name='ds22')(dense21)
dense23 = Dense(256, activation='relu', name='ds23')(dense22)
batch21 = BatchNormalization(name='bt21')(dense23)
dense24 = Dropout(0.3, name='ds24')(batch21)
dense25 = Dense(64, activation='relu', name='ds25')(dense24)
batch22 = BatchNormalization(name='bt22')(dense25)
dense26 = Dropout(0.3, name='ds26')(batch22)
dense27 = Dense(32, activation='relu', name='ds27')(dense26)
dense28 = Dropout(0.3, name='ds28')(dense27)
output2 = Dense(16, activation='softmax', name='ds29')(dense28)

#2-3 모델병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='mg1')                                # dense7와 dense26이 인풋이고 이름을 mg1으로 정의하겠다.
merge2 = Dense(64, activation='relu', name='mg2')(merge1)
merge3 = Dense(128, activation='relu', name='mg3')(merge2)
merge4 = Dense(32, activation='relu', name='mg4')(merge3)
merge5 = Dense(16, activation='relu', name='mg5')(merge4)
last_output = Dense(1, name='last')(merge5)                                         # y가 컬럼이 1개이므로 아웃풋은 1개.

model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')                                                

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                            #   restore_best_weights=False,
                              verbose=1, )

import datetime                                                                 # 데이터 타임 임포트해서 명시해준다.
date = datetime.datetime.now()                                                  # 현재 날짜와 시간이 date로 반환된다.

print(date)                                                                     # 2023-01-12 14:57:54.345489
print(type(date))                                                               # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")                                               # date를 str(문자형)으로 바꾼다.
                                                                                # 0112_1457
print(date)
print(type(date))                                                               # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'                                    # -는 연산 -가 아니라 글자이다. str형이기 때문에...
                                                                                # 0037-0.0048.hdf5

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_weights_only=True,
                      save_best_only=True,
                      # filepath= path + 'MCP/keras52_ModelCheckPoint3.hdf5',
                      filepath = filepath + 'k52_ensemble_sam_amo' + date + '_' + filename
                      )
                                                                        
model.save(path + "keras52_ensemble_sam_amo_save_model_01.h5")

model.fit([x1_train_scale, x2_train_scale], y1_train, epochs=200, batch_size=1, validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)    

#4. 평가, 예측
loss = model.evaluate ([x1_test_scale, x2_test_scale], y1_test)
print('loss : ', loss)

predict = model.predict([x1_test_scale, x2_test_scale])
print('23-01-30 시가 :', predict[-1:],'원') 


"""
23-01-30 시가 : [[65762.89]] 원

230130 09:56
23-01-30 시가 : [[64811.367]] 원

"""