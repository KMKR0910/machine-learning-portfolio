import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # to get the data in the same scale
from sklearn.model_selection import train_test_split # to split the data into training and testing sets
from sklearn import svm #support vector algorithm
from sklearn.metrics import accuracy_score # to check the accuracy 

diabetes_data = pd.read_csv('dataset/diabetes.csv')

print(diabetes_data.head())
print("--------------------------------------------------")

print(diabetes_data.shape)#output the number of rows and columns in the dataset
print("--------------------------------------------------")

print(diabetes_data.describe()) #output the statitic data
print("--------------------------------------------------")

print(diabetes_data['Outcome'].value_counts())#output 1,0 count
print("--------------------------------------------------")


#seperate the data and labels
X= diabetes_data.drop(columns='Outcome')
Y=diabetes_data['Outcome']


scaler= StandardScaler()

scaler.fit(X)

standarddized_data=scaler.transform(X)

X= standarddized_data
Y=diabetes_data['Outcome']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)


classifire =svm.SVC(kernel='linear')

classifire.fit(X_train,Y_train)


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)




#print(diabetes_data.groupby('Outcome').mean())
print("Script executed successfully")