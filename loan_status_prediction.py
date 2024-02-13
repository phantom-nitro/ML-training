import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

loan_dataset = pd.read_csv('Datasets/loan.csv')
print(loan_dataset.head())

#statistical measure of the data
#print(loan_dataset.shape)
#print(loan_dataset.isnull().sum())
print(loan_dataset.describe())

#drop missing values
loan_dataset = loan_dataset.dropna()

#label encoding
loan_dataset.replace({"Loan_Status":{'N':0, 'Y':1}}, inplace=True)
#print(loan_dataset.head())
#print(loan_dataset.isnull().sum())

#Dependent column values
print(loan_dataset["Dependents"].value_counts())
#replace 3+ to 4
loan_dataset.replace({"Dependents":{'3+':4}}, inplace=True)
print(loan_dataset["Dependents"].value_counts())

#Data visualization
'''
sns.countplot(x="Education", hue = "Loan_Status", data = loan_dataset)
plt.show()
sns.countplot(x="Married", hue = "Loan_Status", data = loan_dataset)
plt.show()
'''
#convert categorical columns to numerical values
loan_dataset.replace({"Married":{"No":0,"Yes":1}, "Gender":{"Male":1,"Female":0},"Self_Employed":{"No":0,"Yes":1}, "Property_Area":{"Rural":0,"Semiurban":1,"Urban":2}, "Education":{"Graduate":1, "Not Graduate":0}},inplace=True)
#print(loan_dataset.head())

#separating dependent and independent variable
X = loan_dataset.drop(columns=["Loan_ID", "Loan_Status"], axis=1)
Y = loan_dataset["Loan_Status"]

#print(X)
#print(Y)

#Split train test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1, stratify=Y,random_state=2)

#training the model
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train,Y_train)

#accuracy score on training data
X_train_prediction = classifier.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction,Y_train)
print("train data accuracy: ", train_accuracy)


#accuracy score on test data
X_test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction,Y_test)
print("test data accuracy: ", test_accuracy)


