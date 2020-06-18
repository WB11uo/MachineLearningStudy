import  sklearn.datasets as datasets
from sklearn.neighbors import  KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import  ListedColormap#绘图引入模块

np.random.seed(0)
x=np.sort(5*np.random.rand(40,1),axis=0)
print(x)
y=np.sin(x).ravel()
y[::5]+=1*(0.5-np.random.rand(8))#破坏数据的整齐
T=np.linspace(0,5,100)[:,np.newaxis]
# plt.scatter(x,y)
knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(x,y)#训练数据
newy=knn.predict(T)#预测
plt.scatter(x,y,c='k',label='data')
plt.plot(T,newy,c='g',label='predict')
plt.axis("tight")
plt.legend()
plt.show()