#!/usr/bin/python3
from sklearn.datasets import load_iris

#loading iris data sets
iris=load_iris()

#now splitting into trained and test datasets
from sklearn.model_selection import train_test_split
x,y,z,a=train_test_split(iris.data,iris.target,test_size=0.1)

'''
where x is train_iris{all feature values containing 90%}
      y is remaining test_iris{10% of features}
      z is train_target{all labels containg 90% iris.target}
      a is test_target{remaining 10% of target}
'''
#calling decision tree classifier
from sklearn import tree

dsclf=tree.DecisionTreeClassifier()

#now training data with decision 
trained=dsclf.fit(x,z)

#now time for prediction
output=trained.predict(y)
print(output)

#checking percentage of accuracy
from sklearn.metrics import accuracy_score
check_pct=accuracy_score(a,output)
print(check_pct)

