import pandas as pd
import numpy as np
import matplotlib
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

matplotlib.rcParams['font.sans-serif'] = [u'simHei']
matplotlib.rcParams['axes.unicode_minus'] = False

file_path = '../Data/TCR_all.csv'

data_set = pd.read_csv(file_path, header=None)

data_set['tokens'] = data_set[0].map(lambda x: ' '.join([x[i:i+3] for i in range(len(x)-2)]))
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_set['tokens']).toarray()
X[X>0] = 1
print(X.shape)

data_set['class'] = LabelEncoder().fit_transform(data_set[4])
y = data_set['class'].values

np.save('../Data/matrix_multi.npy', np.column_stack((X, y)))



# 先不降维，获取不同维度下的信息占比
print('Original components: ', X.shape[1])

pca = PCA()
pca.fit_transform(X)


percentiles = {}

# 计算保留信息略高于90%的维度
for remain_info_ in range(90, 100):
    remain_info = float(remain_info_) / 100
    remain_components = np.argwhere(np.array(pca.explained_variance_ratio_).cumsum() > remain_info)[1][0]
    print(f'{remain_info}: {remain_components}')
    percentiles[remain_info_] = remain_components


for percentile in [90]:
    remain_components = percentiles[percentile]

    print(f'{percentile}: {remain_components}')

    # 对X进行降维
    pca = PCA(n_components=remain_components)
    new_X = pca.fit_transform(X)
    # print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    print(new_X.shape)

    np.save('../Data/matrix_multi_90.npy', np.column_stack((new_X, y)))
