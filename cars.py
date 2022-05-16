import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
import graphviz 


names = ['buying', 'maint', 'doors', 'persons', 'lug_boot' , 'safety' , 'class']
df = pd.read_csv ('Alter3.csv' , names=names)

x = df.iloc[:, :-1].values
y = df.iloc[:, 6].values

print(df.head())
print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10) 

#KNN
#
#knn = KNeighborsClassifier(n_neighbors=9)
#knn.fit(X_train,y_train)
#
#print("Preliminary model score:")
#print(knn.score(X_test,y_test))
#
#print(knn.predict([[67,100,100,50,50,0]]))
#
#y_pred = knn.predict(X_test) 
#cm = confusion_matrix(y_test, y_pred)
#print (cm)

#Naive Bayes

#nb_classifier = MultinomialNB()
#
#nb_classifier.fit(X_train, y_train)
#y_pred = nb_classifier.predict(X_test)
#conf_mat = confusion_matrix(y_test, y_pred, labels = ['v-good', 'good', 'acc', 'unacc'])
#print(conf_mat)

#Decision Tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))


