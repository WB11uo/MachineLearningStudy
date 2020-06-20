import  sklearn.datasets as datasets
from sklearn.neighbors import  KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import  ListedColormap#绘图引入模块
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

iris=datasets.load_iris()
x=iris.data
y=iris.target
index=np.arange(150)
np.random.shuffle(index)#打乱数据
# print(index[:-20])
x_train=x[index[:-20]]
y_train=y[index[:-20]]
x_test=x[index[-20:]]
y_test=y[index[-20:]]
tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
score=tree.score(x_test,y_test)
print(score)
tree=KNeighborsClassifier()
tree.fit(x_train,y_train)
score=tree.score(x_test,y_test)
print(score)
# tree=LogisticRegression()
# tree.fit(x_train,y_train)
# score=tree.score(x_test,y_test)
# print(score)
