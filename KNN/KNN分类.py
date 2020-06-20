import  sklearn.datasets as datasets
from sklearn.neighbors import  KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import  ListedColormap#绘图引入模块
iris=datasets.load_iris()
print(iris)
x=iris.data[:,:2]
# print(x)
y=iris.target
#设置 KNN k=15 计算周围临近的15个点
k=15
#图片x,y每一步的步长
h=0.02
#两个颜色分类
cmap_light=ListedColormap(["#FFAAAA","#AAFFAA","#AAAAFF"])#颜色列表
cmap_bold=ListedColormap(["#FF0000","#00FF00","#0000FF"])#颜色列表
myknn=KNeighborsClassifier(n_neighbors=15)
myknn.fit(x,y)#训练数据
print(x)
xmin,xmax=x[:,0].min()-1,x[:,0].max()-1
print(x[:,0])
ymin,ymax=x[:,1].min()-1,x[:,1].max()-1
print(x[:,1])
#生成网格
xx,yy=np.meshgrid(np.arange(xmin,xmax,h),
                  np.arange(ymin,ymax,h)
                  )
print(xx)
#预测
z=myknn.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
#显示背景颜色
plt.pcolormesh(xx,yy,z,cmap=cmap_light)
#显示点的颜色
plt.scatter(x[:,0],x[:,1],c=y,cmap=cmap_bold)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.title("分类")
plt.show()

