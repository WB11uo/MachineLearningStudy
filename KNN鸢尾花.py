import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris=load_iris()
data=iris.data[:,:2]
target=iris.target
label=np.array(target)
index_0=np.where(label==0)
plt.scatter(data[index_0,0],data[index_0,1],marker='x',color='b',label='0',s=15)
index_1=np.where(label==1)
plt.scatter(data[index_1,0],data[index_1,1],marker='o',color='r',label='1',s=15)
index_2=np.where(label==2)
plt.scatter(data[index_2,0],data[index_2,1],marker='s',color='g',label='2',s=15)
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper left')
plt.show()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X,X_test,y_test=train_test_split(data,target,test_size=0.2,random_state=1)
