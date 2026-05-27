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
