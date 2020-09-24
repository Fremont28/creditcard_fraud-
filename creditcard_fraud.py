
#Classify credit card fraud with an isolation forest algorithm 

#import libraries 
import numpy as np 
import csv 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

filename='creditcard 3.csv'
raw_data=open(filename,'rt')
data_credit1=csv.reader(raw_data,delimiter=',')
x=list(data_credit1)
data_credit=np.array(x).astype("float") 

rng = np.random.RandomState(42)

# define features and repsonse 
X=data_credit[:,1:29,] #features 
y=data_credit[:,30] #label 
X_outliers = rng.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
#count binary (0,1) outcomes (fraud or not fraud)
unique, counts = np.unique(y, return_counts=True)
dict(zip(unique,counts))

#train and test data sets 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Isolation forest model 
clf=IsolationForest() 
clf.fit(X_train)

#predict if fraud occurs (0,1) 
pred_train=clf.predict(X_train) #predicting on X_train
y_pred_test=clf.predict(X_test) #predicting on Y_train 
y_pred_outliers=clf.predict(X_outliers)

#count binary (0,1) outcomes (on predictive outcomes)
unique,counts=numpy.unique(y_pred_outliers,return_counts=True)
dict(zip(unique,counts))



