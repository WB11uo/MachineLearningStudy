import numpy
import matplotlib
import sklearn

from sklearn.neighbors import  KNeighborsClassifier #KNN分类器

x_train=[[185,80,43],[170,70,41],[163,45,36],
        [168,48,37],[156,41,35]]
y_train=["男","男","男","女","女"]

knn=KNeighborsClassifier(n_neighbors=3) #创建机器学习的KNN对象

knn.fit(x_train,y_train) #训练数据，自适应数据简历数据模型

Test_data=[[182,75,41],[159,46,37]] #随机数据测试

target=knn.predict(Test_data)
print(target)






