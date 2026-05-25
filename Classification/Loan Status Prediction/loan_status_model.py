import numpy as np
import pandas as pd  
import os
import seaborn as sns # for data visualization
from sklearn.model_selection import train_test_split # for splitting the data into training and testing sets
from sklearn import svm
from sklearn.metrics import accuracy_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(BASE_DIR, "dataset", "loan_status.csv")

loan_dataset = pd.read_csv(file_path)




print(type(loan_dataset))

print(loan_dataset.shape)
print(loan_dataset.head())