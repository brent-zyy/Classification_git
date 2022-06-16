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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['font.sans-serif'] = [u'simHei']
matplotlib.rcParams['axes.unicode_minus'] = False

file_path = '../Data/matrix_multi_90.npy'
data = np.load(file_path)

# file_path = './Data/matrix_multi.npy'
# data = np.load(file_path)

y = data[:, -1]
X = data[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(X, y)

print("start")

clf=tree.DecisionTreeClassifier()        #1-1定义一种二分类算法
ovo=OneVsOneClassifier(clf)    #1-2进行多分类转换OVO
ovo.fit(X_train,y_train)            #1-3进行数据训练与预测
print(ovo.score(X_test,y_test))
print("==========================================")

clf=LinearDiscriminantAnalysis()        #1-1定义一种二分类算法
ovo=OneVsOneClassifier(clf)    #1-2进行多分类转换OVO
ovo.fit(X_train,y_train)            #1-3进行数据训练与预测
print(ovo.score(X_test,y_test))
print("==========================================")

clf=MultinomialNB(alpha=0.01)        #1-1定义一种二分类算法
ovo=OneVsOneClassifier(clf)    #1-2进行多分类转换OVO
ovo.fit(X_train,y_train)            #1-3进行数据训练与预测
print(ovo.score(X_test,y_test))
print("==========================================")

clf=KNeighborsClassifier()        #1-1定义一种二分类算法
ovo=OneVsOneClassifier(clf)    #1-2进行多分类转换OVO
ovo.fit(X_train,y_train)            #1-3进行数据训练与预测
print(ovo.score(X_test,y_test))
print("==========================================")

clf=SVC(kernel='rbf', probability=True)        #1-1定义一种二分类算法
ovo=OneVsOneClassifier(clf)    #1-2进行多分类转换OVO
ovo.fit(X_train,y_train)            #1-3进行数据训练与预测
print(ovo.score(X_test,y_test))
print("==========================================")