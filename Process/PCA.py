from sklearn.decomposition import PCA, KernelPCA
import pandas as pd
import numpy as np

file_path = '../Data/matrix.csv'
data_set = pd.read_csv(file_path, dtype=np.int8)
data = data_set.values

y = data[:, -1]
X = data[:, :-1]

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

    np.save(file_path.replace('.csv', f'-{percentile}.npy'), np.column_stack((new_X, y)))