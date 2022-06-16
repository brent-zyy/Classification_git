import time
from sklearn import tree
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_predict
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = [u'simHei']
matplotlib.rcParams['axes.unicode_minus'] = False

#data_set = pd.read_csv('./Data/matrix.csv')
#data = data_set.values

file_path = '../Data/matrix_90.npy'
data = np.load(file_path)

y = data[:, -1]
X = data[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(X, y)


clf = SVC(kernel='rbf', probability=True)
time_start = time.time()
clf.fit(X_train, y_train)
print(f'time for train: {time.time() - time_start}')
time_start = time.time()
predictions = cross_val_predict(clf, X, y, cv=5)
print(f'time for predict(per fold): {(time.time() - time_start)/5}')

print("SVC")
print(classification_report(y, predictions))
print("AC", accuracy_score(y, predictions))