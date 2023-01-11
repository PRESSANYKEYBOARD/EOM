from sklearn import datasets
print(datasets.get_data_home())
# C:\Users\EOMNotebook\scikit_learn-data                        # scikit_learn-data 다운로드 받은 장소
from sklearn.datasets import fetch_covtype, load_wine