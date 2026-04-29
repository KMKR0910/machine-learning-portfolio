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

#data standardization - in this conevert data into 0,1 scale
scaler= StandardScaler()

scaler.fit(X)

standarddized_data=scaler.transform(X)

X= standarddized_data
Y=diabetes_data['Outcome']

print(X)
print(Y)

#split data (test data=20%,stratify-get same distribution both data,randomstate-get same data every time)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)


#model training(supprot vector machine)

classifire =svm.SVC(kernel='linear')

classifire.fit(X_train,Y_train)

X_train_prediction=classifire.predict(X_train) #predict the output for training data-in this simply predcit y value of x and next , check with real y
trainig_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy score:", trainig_data_accuracy)

X_test_prediction=classifire.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy score for test data:", testing_data_accuracy)




#prediction
input_data = (4,110,92,0,0,37.6,0.191,30)#get new data

numpy_array_data=np.asarray(input_data)

reshaped_data=numpy_array_data.reshape(1,-1)#for reshaping  data because predicting for one data and it should be in 2D array

std_data=scaler.transform(reshaped_data)


prediction=classifire.predict(std_data)#get output
print(prediction)

if prediction[0]==0:
    print("The person is not diabetic")   
else :
      print("The person is diabetic")



#print(diabetes_data.groupby('Outcome').mean())
print("Script executed successfully")