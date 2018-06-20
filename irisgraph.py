#!/usr/bin/env python3
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#loading iris data sets
iris=load_iris()

#now splitting into trained and test datasets
train_iris,test_iris,train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.2)

# team_iris &train_target data will be using under fit method
#calling decision tree classifier

clf=tree.DecisionTreeClassifier()

#now training data with decision 
trained=clf.fit(train_iris,train_target)

#test with test_iris
output=trained.predict(test_iris)
print(output)

#actual output
print(test_target)

#calling accuracy score
pct=accuracy_score(test_target,output)
print(pct)

#exporting graph for decision tree
tree.export_graphviz(clf, out_file="tree.dot", max_depth=7, feature_names=iris.feature_names, class_names=iris.target_names, filled=True,rounded=True)
