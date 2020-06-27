import  numpy as np
import matplotlib.pyplot as plt
from sklearn import  datasets
boston=datasets.load_boston()
x=boston.data[:,5]
y=boston.target
x=x[y<50]
y=y[y<50]
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.1)
model=LinearRegression()
model.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))