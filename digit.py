#!/usr/bin/python3
from sklearn.datasets import load_digits
import numpy as np
from sklearn.svm import SVC 
import matplotlib.pyplot as plt

''' 
to find datasets in scikit-learn
for i in dir(datasets):
   if digits in i
    print(i)
'''

#loading digit images
digit=load_digits()

#only features data
training_data=digit.data

#only target data
training_target=digit.target

#training data extract from original data
td_original=np.delete(training_data,-1,axis=0)

#training data extract from original target data
tt_original=np.delete(training_target,-1)

#calling SVC
clf=SVC()

#training algo
trained= clf.fit(td_original,tt_original)

#now time for prediction
output=trained.predict(digit.data[-2].reshape(1,64))
print(output)

#plotting that testing image
plt.imshow(digit.images[-2])
plt.show()

