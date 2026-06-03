import numpy as np
import pandas as pd  
import os
import seaborn as sns # for data visualization
from sklearn.model_selection import train_test_split # for splitting the data into training and testing sets
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(BASE_DIR, "dataset", "loan_status.csv")

loan_dataset = pd.read_csv(file_path)




print(type(loan_dataset))

print(loan_dataset.shape)
print(loan_dataset.head())

print(loan_dataset.describe())#for statistical data

print(loan_dataset.isnull().sum())

loan_dataset = loan_dataset.dropna() #for drop missing values
print(loan_dataset.isnull().sum())

loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}}, inplace=True)
print(loan_dataset.head())

print(loan_dataset['Dependents'].value_counts())

loan_dataset.replace({"Dependents":{'3+':'4'}}, inplace=True)

print(loan_dataset['Dependents'].value_counts())



sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
plt.title("Education vs Loan Status")
plt.xlabel("Education")
plt.ylabel("Count")

plt.show()


sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)
plt.title("Married vs Loan Status")
plt.xlabel("Married")   
plt.ylabel("Count")

plt.show()

loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

loan_dataset.head()

X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']

X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear')

#training the support Vector Macine model
classifier.fit(X_train,Y_train)

# accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuray = accuracy_score(X_train_prediction,Y_train)

print('Accuracy on training data : ', training_data_accuray)

# accuracy score on training data
X_test_prediction = classifier.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,Y_test)

print('Accuracy on test data : ', test_data_accuray)