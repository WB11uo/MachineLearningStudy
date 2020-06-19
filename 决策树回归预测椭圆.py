import  sklearn.datasets as datasets
from sklearn.neighbors import  KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import  ListedColormap#绘图引入模块
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

x=200*np.random.rand(100,1)-100
x_train=np.sort(x,axis=0)
y_train=np.pi*np.array([np.sin(x_train).ravel(),np.cos(x_train).ravel()])
y_train=y_train.transpose()
# plt.scatter(y_train[:,0],y_train[:,1])
# plt.axis("equal")
# plt.show()
y_train[::5]+=np.random.rand(20,2)#插入干扰数据
tree1=DecisionTreeRegressor(max_depth=5)
tree2=DecisionTreeRegressor(max_depth=10)
tree3=DecisionTreeRegressor(max_depth=50) #三个决策树回归
tree1.fit(x_train,y_train)
tree2.fit(x_train,y_train)
tree3.fit(x_train,y_train)
x_test=np.arange(-100,100,0.1).reshape((-1,1))
y_new1=tree1.predict(x_test)
y_new2=tree2.predict(x_test)
y_new3=tree3.predict(x_test)
fig=plt.figure(figsize=(12,9))
axes1=fig.add_subplot(221)
s1=axes1.scatter(y_train[:,0],y_train[:,1],label="orignal")
axes1.legend()
axes2=fig.add_subplot(222)
s2=axes2.scatter(y_new1[:,0],y_new1[:,1],label="depth=5")
axes2.legend()
axes3=fig.add_subplot(223)
s3=axes3.scatter(y_new2[:,0],y_new2[:,1],label="depth=10")
axes3.legend()
axes4=fig.add_subplot(224)
s4=axes4.scatter(y_new3[:,0],y_new3[:,1],label="depth=50")
axes4.legend()
fig.legend((s1,s2,s3,s4),(1,2,3,4),("upper right"))
plt.show()