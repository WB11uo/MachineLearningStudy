from sklearn import datasets
from sklearn.naive_bayes import  GaussianNB
from sklearn.naive_bayes import  MultinomialNB
iris=datasets.load_iris()
x=iris.data
y=iris.target
gnb=GaussianNB()
a=gnb.fit(x,y).score(x,y)
bnb=MultinomialNB()
b=bnb.fit(x,y).score(x,y)



