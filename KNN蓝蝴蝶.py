import  sklearn.datasets as datasets
from sklearn.neighbors import  KNeighborsClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import  ListedColormap#绘图引入模块

iris=datasets.load_iris()#蓝蝴蝶
# print(iris)
x_train=iris.data[::2]#Python序列切片地址可以写为[开始：结束：步长]，其中的开始和结束可以省略
# print(x_train)
y_train=iris.target[::2]

x_test=iris.data[1::2]
y_test=iris.target[1::2]

knn=KNeighborsClassifier()#KNN分类器
knn.fit(x_train,y_train) #训练数据
y_=knn.predict(x_test)#数据预测
score=knn.score(x_test,y_test)#y_test换成y_结果为1.0
print(score)

cmap=ListedColormap(["#FF0000","#00FF00","#0000FF"])#颜色列表
plt.scatter(iris.data[:,2],iris.data[:,3],c=iris.target)#c未指定具体颜色，自动绘图
print(iris.data)
print(iris.data[:,2])
print(iris.data[:,3])
plt.plot(x_test,y_test)
plt.show()

